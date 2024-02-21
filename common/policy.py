from common.activation import FTA
import common.builder as builder
import common.distribution as distribution
import common.util as util
from common.feature_extractor import TCN

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

try:
    from functorch import combine_state_for_ensemble, vmap
except:
    print("no functorh")

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, obs):
        raise NotImplementedError


class GaussianPolicyBuilder(nn.Module):
    """https://github.com/pairlab/d2rl/blob/main/sac/model.py"""

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=32,
        num_layers=2,
        resnet=False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.resnet = resnet

        in_dim = hidden_dim + observation_dim if resnet else hidden_dim

        self.linear1 = nn.Linear(observation_dim, hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 2:
            self.linear3 = nn.Linear(in_dim, hidden_dim)
            self.linear4 = nn.Linear(in_dim, hidden_dim)
        if num_layers > 4:
            self.linear5 = nn.Linear(in_dim, hidden_dim)
            self.linear6 = nn.Linear(in_dim, hidden_dim)
        if num_layers == 8:
            self.linear7 = nn.Linear(in_dim, hidden_dim)
            self.linear8 = nn.Linear(in_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)
        nn.init.xavier_uniform_(self.mean_linear.weight, 1e-3)

    def forward(self, state):
        x = F.selu(self.linear1(state))
        x = torch.cat([x, state], dim=1) if self.resnet else x

        x = F.selu(self.linear2(x))

        if self.num_layers > 2:
            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = F.selu(self.linear3(x))

            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = F.selu(self.linear4(x))

        if self.num_layers > 4:
            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = F.selu(self.linear5(x))

            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = F.selu(self.linear6(x))

        if self.num_layers == 8:
            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = F.selu(self.linear7(x))

            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = F.selu(self.linear8(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return mean, log_std


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=32,
        num_layers=2,
        resnet=False,
    ) -> None:
        super().__init__()
        self.model = torch.jit.trace(
            GaussianPolicyBuilder(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                resnet=resnet,
            ),
            example_inputs=(torch.rand(1, observation_dim)),
        )

    def sample(self, obs):
        means, log_stds = self.forward(obs)
        normals, xs, actions = self._get_distribution(means, log_stds)
        entropy = self._calc_entropy(normals, xs, actions)
        return actions, entropy, means

    def forward(self, state):
        mean, log_std = self.model(state)
        return mean, log_std

    def _get_distribution(self, means, log_stds):
        stds = log_stds.exp()
        normals = Normal(means, stds)
        xs = normals.rsample()
        actions = torch.tanh(xs)
        return normals, xs, actions

    def _calc_entropy(self, normals, xs, actions, dim=1):
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropy = -log_probs.sum(dim=dim, keepdim=True)
        return entropy


class MultiheadGaussianPolicyBuilder(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=32,
        num_layers=4,
        resnet=False,
        layernorm=False,
        fta=False,
        fta_delta=0.2,
        max_nheads=int(100),
    ):
        super().__init__()
        self.num_layers = num_layers
        self.max_nheads = max_nheads
        self.action_dim = action_dim

        self.resnet = resnet
        self.layernorm = layernorm
        self.fta = fta

        in_dim = hidden_dim + observation_dim if resnet else hidden_dim
        out_dim = action_dim * self.max_nheads

        if self.layernorm:
            self.ln1 = nn.LayerNorm(in_dim)
            self.ln2 = nn.LayerNorm(in_dim)
            if num_layers > 2:
                self.ln3 = nn.LayerNorm(in_dim)
                self.ln4 = nn.LayerNorm(in_dim)
            if num_layers > 4:
                self.ln5 = nn.LayerNorm(in_dim)
                self.ln6 = nn.LayerNorm(in_dim)
            if num_layers == 8:
                self.ln7 = nn.LayerNorm(in_dim)

        self.linear1 = nn.Linear(observation_dim, hidden_dim)
        self.linear2 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 2:
            self.linear3 = nn.Linear(in_dim, hidden_dim)
            self.linear4 = nn.Linear(in_dim, hidden_dim)
        if num_layers > 4:
            self.linear5 = nn.Linear(in_dim, hidden_dim)
            self.linear6 = nn.Linear(in_dim, hidden_dim)
        if num_layers == 8:
            self.linear7 = nn.Linear(in_dim, hidden_dim)
            self.linear8 = nn.Linear(in_dim, hidden_dim)

        if fta:
            self.ln_l = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.fta_l = FTA(delta=fta_delta)

            hidden_dim *= self.fta_l.nbins

        self.mean_linear = nn.Linear(hidden_dim, out_dim)
        self.log_std_linear = nn.Linear(hidden_dim, out_dim)

        self.apply(weights_init_)
        nn.init.xavier_uniform_(self.mean_linear.weight, 1e-3)

    def forward(self, state):
        x = F.selu(self.linear1(state))

        x = torch.cat([x, state], dim=1) if self.resnet else x
        x = self.ln1(x) if self.layernorm else x
        x = F.selu(self.linear2(x))

        if self.num_layers > 2:
            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = self.ln2(x) if self.layernorm else x
            x = F.selu(self.linear3(x))

            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = self.ln3(x) if self.layernorm else x
            x = F.selu(self.linear4(x))

        if self.num_layers > 4:
            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = self.ln4(x) if self.layernorm else x
            x = F.selu(self.linear5(x))

            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = self.ln5(x) if self.layernorm else x
            x = F.selu(self.linear6(x))

        if self.num_layers == 8:
            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = self.ln6(x) if self.layernorm else x
            x = F.selu(self.linear7(x))

            x = torch.cat([x, state], dim=1) if self.resnet else x
            x = self.ln7(x) if self.layernorm else x
            x = F.selu(self.linear8(x))

        if self.fta:
            x = self.ln_l(x)
            x = self.fta_l(x)

        means = self.mean_linear(x).view(-1, self.max_nheads, self.action_dim)
        log_stds = self.log_std_linear(x).view(-1, self.max_nheads, self.action_dim)
        log_stds = torch.clamp(log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return means, log_stds


class MultiheadGaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    eps = 1e-6

    def __init__(
        self,
        observation_dim,
        action_dim,
        n_heads,
        hidden_dim=32,
        num_layers=2,
        resnet=False,
        layernorm=False,
        fta=False,
        fta_delta=0.25,
        max_nheads=int(50),
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.max_nheads = max_nheads
        self.action_dim = action_dim

        self.model = torch.jit.trace(
            MultiheadGaussianPolicyBuilder(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                resnet=resnet,
                layernorm=layernorm,
                fta=fta,
                fta_delta=fta_delta,
                max_nheads=max_nheads,
            ),
            example_inputs=(torch.rand(1, observation_dim)),
        )

    def sample(self, obs):
        means, log_stds = self.forward(obs)  # [N, H, A], [N, H, A]
        normals, xs, actions = self._get_distribution(means, log_stds)
        entropies = self._calc_entropy(normals, xs, actions, dim=2)  # [N, H, 1]
        return actions, entropies, normals, means  # [N, H, A], [N, H, 1], [N, H, A]

    def _sample(self, obs):
        means, log_stds = self.forward(obs)  # [N, H, A], [N, H, A]
        normals, xs, actions = self._get_distribution(means, log_stds)
        entropies = self._calc_entropy(normals, xs, actions, dim=2)  # [N, H, 1]
        return actions, entropies, means  # [N, H, A], [N, H, 1], [N, H, A]

    def forward(self, state):
        mean, log_std = self.model(state)
        return mean[:, : self.n_heads, :], log_std[:, : self.n_heads, :]

    def add_head(self, n_heads=1):
        self.n_heads += n_heads
        assert (
            self.n_heads <= self.max_nheads
        ), f"exceed max num heads {self.max_nheads}"

    def _get_distribution(self, means, log_stds):
        stds = log_stds.exp()
        normals = Normal(means, stds)
        xs = normals.rsample()
        actions = torch.tanh(xs)
        return normals, xs, actions

    def _calc_entropy(self, normals, xs, actions, dim: int = 1):
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropy = -log_probs.sum(dim=dim, keepdim=True)
        return entropy


class DynamicMultiheadGaussianPolicy(BaseNetwork):
    """Warning: not scriptable and the performance is poor"""

    def __init__(
        self,
        observation_dim,
        action_dim,
        n_heads,
        sizes=[64, 64],
        squash=True,
        activation="relu",
        layernorm=False,
        fuzzytiling=False,
        initializer="xavier_uniform",
        device="cpu",
    ):
        super().__init__()
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.eps = 1e-6

        self.device = device
        self.init_nheads = n_heads
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash
        self.tanh = nn.Tanh() if squash else None

        base_model, base_out_dim = builder.create_linear_base(
            model=[],
            units=observation_dim,
            hidden_units=sizes,
            hidden_activation=activation,
            layernorm=layernorm,
        )

        if fuzzytiling:
            base_model.pop()
            fta = FTA()
            base_model.append(fta)
            base_out_dim *= fta.nbins

        self.base_out_dim = base_out_dim
        self.base_model = (
            nn.Sequential(*base_model)
            .apply(builder.initialize_weights(builder.str_to_initializer[initializer]))
            .to(self.device)
        )

        self.n_heads = 0
        self.heads = nn.ModuleList([])
        self.params = None
        self.add_head(n_heads)

    def sample(self, obs):
        means, log_stds = self.forward(obs)
        normals, xs, actions = self._get_distribution(means, log_stds)
        entropies = self._calc_entropy(normals, xs, actions, dim=2)  # [N, H, 1]
        return actions, entropies, means  # [N, H, A], [N, H, 1], [N, H, A]

    def forward(self, obs):
        """forward multi-head"""
        x = self.base_model(obs)
        x = self.heads_model(x)  # [H, N, 2*A]
        x = x.view([-1, self.n_heads, 2 * self.action_dim])  # [N, H, 2*A]
        means, log_stds = self._calc_mean_std(x)  # [N, H, A], [N, H, A]
        return means, log_stds

    def forward_head(self, obs, idx):
        """forward single head"""
        x = self.base_model(obs)
        x = self.heads[idx](x)
        means, log_stds = self._calc_mean_std(x)  # [N, A]
        normals, xs, actions = self._get_distribution(means, log_stds)
        entropies = self._calc_entropy(normals, xs, actions, dim=1)  # [N, 1]
        return actions, entropies, means

    def add_head(self, n_heads=1):
        for _ in range(n_heads):
            head = nn.Linear(self.base_out_dim, 2 * self.action_dim).to(self.device)
            nn.init.xavier_uniform_(head.weight, 1e-3)

            self.heads.append(head)
            self.n_heads += 1

        self._ensemble()

    def _calc_mean_std(self, x):
        means, log_stds = torch.chunk(
            x, 2, dim=-1
        )  # [N, H, A], [N, H, A] <-- [N, H, 2A]
        log_stds = torch.clamp(log_stds, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)
        return means, log_stds

    def _get_distribution(self, means, log_stds):
        stds = log_stds.exp()
        normals = Normal(means, stds)
        xs = normals.rsample()
        actions = self.tanh(xs) if self.squash else xs
        return normals, xs, actions

    def _calc_entropy(self, normals, xs, actions, dim: int = 1):
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropy = -log_probs.sum(dim=dim, keepdim=True)
        return entropy

    def _ensemble(self):
        if self.params is not None:
            self._update_heads_param()

        fmodel, self.params, bufs = combine_state_for_ensemble(self.heads)
        [p.requires_grad_().to(self.device) for p in self.params]

        self.heads_model = lambda x: (vmap(fmodel, in_dims=(0, 0, None)))(
            self.params, bufs, x
        )

    def _update_heads_param(self):
        for i in range(self.n_heads):
            self.heads[i].weight.data = self.params[0][i].data
            self.heads[i].bias.data = self.params[1][i].data


class GaussianMixturePolicy(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[64, 64],
        n_gauss=10,
        reg=0.001,
        reparameterize=True,
    ) -> None:
        super().__init__()
        self.eps = 1e-2

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.n_gauss = n_gauss
        self.reg = reg
        self.reparameterize = reparameterize

        self.model = distribution.GaussianMixture(
            input_dim=self.observation_dim,
            output_dim=self.action_dim,
            hidden_layers_sizes=sizes,
            K=n_gauss,
            reg=reg,
            reparameterize=reparameterize,
        )

    def forward(self, obs):
        act, logp, mean = self.model(obs)
        act = torch.tanh(act)
        mean = torch.tanh(mean)
        logp -= self.squash_correction(act)
        entropy = -logp[:, None].sum(dim=1, keepdim=True)
        return act, entropy, mean

    def squash_correction(self, inp):
        return torch.sum(torch.log(1 - torch.tanh(inp) ** 2 + self.eps), 1)

    def reg_loss(self):
        return self.model.reg_loss_t


class StochasticPolicy(BaseNetwork):
    """Stochastic NN policy"""

    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[64, 64],
        squash=True,
        layernorm=False,
        activation="relu",
        initializer="xavier_uniform",
        device="cpu",
    ):
        super().__init__()
        self.device = device

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash

        output_activation = "tanh" if squash else None
        self.model = builder.create_linear_network(
            input_dim=self.observation_dim + self.action_dim,
            output_dim=action_dim * 2,
            hidden_units=sizes,
            hidden_activation=activation,
            output_activation=output_activation,
            layernorm=layernorm,
            initializer=initializer,
        )
        nn.init.xavier_uniform_(self.model[-2].weight, 1e-3)

    def forward(self, x):
        return self.model(x)

    def sample(self, obs):
        acts = self.get_actions(obs)
        a = acts[:, 0, :]  # TODO: add some randomness to explorative action
        return a, 0, a  # TODO: compute entropy

    def get_action(self, obs):
        return self.get_actions(obs).squeeze(0)[0]

    def get_actions(self, obs, n_act=1):
        obs = util.check_obs(obs)
        n_obs = obs.shape[0]

        latent_shape = (n_act, self.action_dim)
        latents = torch.normal(0, 1, size=latent_shape).to(self.device)

        s, a = util.get_sa_pairs(obs, latents)
        raw_actions = self.forward(torch.cat([s, a], -1)).view(
            n_obs, n_act, self.action_dim
        )

        return raw_actions


if __name__ == "__main__":
    from torch.profiler import ProfilerActivity, profile, record_function

    obs_dim = 5
    act_dim = 2
    n_heads = 20
    hidden_dim = 32
    num_layers = 4

    fta = True
    resnet = True
    layernorm = True

    device = "cuda"

    times = 100
    obs = torch.rand(100, obs_dim).to(device)

    policy1 = MultiheadGaussianPolicy(
        observation_dim=obs_dim,
        action_dim=act_dim,
        n_heads=n_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        resnet=resnet,
        layernorm=layernorm,
        fta=fta,
    ).to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof1:
        with record_function("model_inference"):
            for _ in range(times):
                policy1.sample(obs)

    print(prof1.key_averages().table(sort_by="cuda_time_total", row_limit=10))
