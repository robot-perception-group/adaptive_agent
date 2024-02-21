import torch
import torch.nn as nn
import torch.nn.functional as F

from common.activation import FTA
from common.util import check_samples


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


class QNetworkBuilder(BaseNetwork):
    """https://github.com/pairlab/d2rl/blob/main/sac/model.py"""

    def __init__(self, observation_dim, action_dim, hidden_dim, num_layers):
        super().__init__()

        in_dim = observation_dim + action_dim + hidden_dim
        # Q1 architecture
        self.l1_1 = nn.Linear(observation_dim + action_dim, hidden_dim)
        self.l1_2 = nn.Linear(in_dim, hidden_dim)

        self.l2_1 = nn.Linear(observation_dim + action_dim, hidden_dim)
        self.l2_2 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 2:
            self.l1_3 = nn.Linear(in_dim, hidden_dim)
            self.l1_4 = nn.Linear(in_dim, hidden_dim)

            self.l2_3 = nn.Linear(in_dim, hidden_dim)
            self.l2_4 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 4:
            self.l1_5 = nn.Linear(in_dim, hidden_dim)
            self.l1_6 = nn.Linear(in_dim, hidden_dim)

            self.l2_5 = nn.Linear(in_dim, hidden_dim)
            self.l2_6 = nn.Linear(in_dim, hidden_dim)

        if num_layers == 8:
            self.l1_7 = nn.Linear(in_dim, hidden_dim)
            self.l1_8 = nn.Linear(in_dim, hidden_dim)

            self.l2_7 = nn.Linear(in_dim, hidden_dim)
            self.l2_8 = nn.Linear(in_dim, hidden_dim)

        self.out1 = nn.Linear(hidden_dim, 1)
        self.out2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        self.num_layers = num_layers

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.l1_1(xu))
        x2 = F.relu(self.l2_1(xu))
        x1 = torch.cat([x1, xu], dim=1)
        x2 = torch.cat([x2, xu], dim=1)

        x1 = F.relu(self.l1_2(x1))
        x2 = F.relu(self.l2_2(x2))
        if not self.num_layers == 2:
            x1 = torch.cat([x1, xu], dim=1)
            x2 = torch.cat([x2, xu], dim=1)

        if self.num_layers > 2:
            x1 = F.relu(self.l1_3(x1))
            x2 = F.relu(self.l2_3(x2))
            x1 = torch.cat([x1, xu], dim=1)
            x2 = torch.cat([x2, xu], dim=1)

            x1 = F.relu(self.l1_4(x1))
            x2 = F.relu(self.l2_4(x2))
            if not self.num_layers == 4:
                x1 = torch.cat([x1, xu], dim=1)
                x2 = torch.cat([x2, xu], dim=1)

        if self.num_layers > 4:
            x1 = F.relu(self.l1_5(x1))
            x2 = F.relu(self.l2_5(x2))
            x1 = torch.cat([x1, xu], dim=1)
            x2 = torch.cat([x2, xu], dim=1)

            x1 = F.relu(self.l1_6(x1))
            x2 = F.relu(self.l2_6(x2))
            if not self.num_layers == 6:
                x1 = torch.cat([x1, xu], dim=1)
                x2 = torch.cat([x2, xu], dim=1)

        if self.num_layers == 8:
            x1 = F.relu(self.l1_7(x1))
            x2 = F.relu(self.l2_7(x2))
            x1 = torch.cat([x1, xu], dim=1)
            x2 = torch.cat([x2, xu], dim=1)

            x1 = F.relu(self.l1_8(x1))
            x2 = F.relu(self.l2_8(x2))

        x1 = self.out1(x1)
        x2 = self.out2(x2)

        return x1, x2


class QNetwork(BaseNetwork):
    def __init__(
        self, observation_dim, action_dim, hidden_dim=64, num_layers=2
    ) -> None:
        super().__init__()
        self.model = torch.jit.trace(
            QNetworkBuilder(observation_dim, action_dim, hidden_dim, num_layers),
            example_inputs=(
                torch.rand(1, observation_dim),
                torch.rand(1, action_dim),
            ),
        )

    def forward(self, state, action):
        x1, x2 = self.model(state, action)
        return x1, x2


class MultiheadSFNetworkBuilder(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        hidden_dim,
        num_layers,
        resnet=False,
        layernorm=False,
        fta=False,
        fta_delta=0.2,
        max_nheads=int(100),
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.max_nheads = max_nheads

        self.feature_dim = feature_dim

        self.resnet = resnet
        self.layernorm = layernorm
        self.fta = fta

        in_dim = observation_dim + action_dim + hidden_dim if resnet else hidden_dim
        out_dim = feature_dim * self.max_nheads

        if self.layernorm:
            self.ln1_1 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_1 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_2 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_2 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_3 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_3 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_4 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_4 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_5 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_5 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_6 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_6 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_7 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_7 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln1_8 = nn.LayerNorm(in_dim, elementwise_affine=True)
            self.ln2_8 = nn.LayerNorm(in_dim, elementwise_affine=True)

        if fta:
            self.fta_ln = nn.LayerNorm(
                observation_dim + action_dim, elementwise_affine=False
            )
            self.fta = FTA(delta=fta_delta)

            next_dim = (observation_dim + action_dim) * self.fta.nbins
            self.fta_l = nn.Linear(next_dim, observation_dim + action_dim)

        self.l1_1 = nn.Linear(observation_dim + action_dim, hidden_dim)
        self.l2_1 = nn.Linear(observation_dim + action_dim, hidden_dim)

        self.l1_2 = nn.Linear(in_dim, hidden_dim)
        self.l2_2 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 2:
            self.l1_3 = nn.Linear(in_dim, hidden_dim)
            self.l1_4 = nn.Linear(in_dim, hidden_dim)

            self.l2_3 = nn.Linear(in_dim, hidden_dim)
            self.l2_4 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 4:
            self.l1_5 = nn.Linear(in_dim, hidden_dim)
            self.l1_6 = nn.Linear(in_dim, hidden_dim)

            self.l2_5 = nn.Linear(in_dim, hidden_dim)
            self.l2_6 = nn.Linear(in_dim, hidden_dim)

        if num_layers == 8:
            self.l1_7 = nn.Linear(in_dim, hidden_dim)
            self.l1_8 = nn.Linear(in_dim, hidden_dim)

            self.l2_7 = nn.Linear(in_dim, hidden_dim)
            self.l2_8 = nn.Linear(in_dim, hidden_dim)

        self.out1 = nn.Linear(hidden_dim, out_dim)
        self.out2 = nn.Linear(hidden_dim, out_dim)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        if self.fta:
            xu = self.fta_ln(xu)
            xu = self.fta(xu)
            xu = self.fta_l(xu)

        x1 = F.relu(self.l1_1(xu))
        x2 = F.relu(self.l2_1(xu))

        x1 = torch.cat([x1, xu], dim=1) if self.resnet else x1
        x2 = torch.cat([x2, xu], dim=1) if self.resnet else x2

        x1 = self.ln1_1(x1) if self.layernorm else x1
        x2 = self.ln2_1(x2) if self.layernorm else x2

        x1 = F.relu(self.l1_2(x1))
        x2 = F.relu(self.l2_2(x2))

        if not self.num_layers == 2:
            x1 = torch.cat([x1, xu], dim=1) if self.resnet else x1
            x2 = torch.cat([x2, xu], dim=1) if self.resnet else x2

            x1 = self.ln1_2(x1) if self.layernorm else x1
            x2 = self.ln2_2(x2) if self.layernorm else x2

        if self.num_layers > 2:
            x1 = F.relu(self.l1_3(x1))
            x2 = F.relu(self.l2_3(x2))

            x1 = torch.cat([x1, xu], dim=1) if self.resnet else x1
            x2 = torch.cat([x2, xu], dim=1) if self.resnet else x2

            x1 = self.ln1_3(x1) if self.layernorm else x1
            x2 = self.ln2_3(x2) if self.layernorm else x2

            x1 = F.relu(self.l1_4(x1))
            x2 = F.relu(self.l2_4(x2))
            if not self.num_layers == 4:
                x1 = torch.cat([x1, xu], dim=1) if self.resnet else x1
                x2 = torch.cat([x2, xu], dim=1) if self.resnet else x2

                x1 = self.ln1_4(x1) if self.layernorm else x1
                x2 = self.ln2_4(x2) if self.layernorm else x2

        if self.num_layers > 4:
            x1 = F.relu(self.l1_5(x1))
            x2 = F.relu(self.l2_5(x2))

            x1 = torch.cat([x1, xu], dim=1) if self.resnet else x1
            x2 = torch.cat([x2, xu], dim=1) if self.resnet else x2

            x1 = self.ln1_5(x1) if self.layernorm else x1
            x2 = self.ln2_5(x2) if self.layernorm else x2

            x1 = F.relu(self.l1_6(x1))
            x2 = F.relu(self.l2_6(x2))

            if not self.num_layers == 6:
                x1 = torch.cat([x1, xu], dim=1) if self.resnet else x1
                x2 = torch.cat([x2, xu], dim=1) if self.resnet else x2

                x1 = self.ln1_6(x1) if self.layernorm else x1
                x2 = self.ln2_6(x2) if self.layernorm else x2

        if self.num_layers == 8:
            x1 = F.relu(self.l1_7(x1))
            x2 = F.relu(self.l2_7(x2))

            x1 = torch.cat([x1, xu], dim=1) if self.resnet else x1
            x2 = torch.cat([x2, xu], dim=1) if self.resnet else x2

            x1 = self.ln1_7(x1) if self.layernorm else x1
            x2 = self.ln2_7(x2) if self.layernorm else x2

            x1 = F.relu(self.l1_8(x1))
            x2 = F.relu(self.l2_8(x2))

        x1 = self.out1(x1).view(-1, self.max_nheads, self.feature_dim)
        x2 = self.out2(x2).view(-1, self.max_nheads, self.feature_dim)

        return x1, x2


class MultiheadSFNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        n_heads,
        hidden_dim=64,
        num_layers=4,
        resnet=False,
        layernorm=False,
        fta=False,
        fta_delta=0.25,
        max_nheads=int(100),
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.max_nheads = max_nheads

        self.model = torch.jit.trace(
            MultiheadSFNetworkBuilder(
                observation_dim=observation_dim,
                feature_dim=feature_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                resnet=resnet,
                layernorm=layernorm,
                fta=fta,
                fta_delta=fta_delta,
                max_nheads=max_nheads,
            ),
            example_inputs=(
                torch.rand(1, observation_dim),
                torch.rand(1, action_dim),
            ),
        )

    def forward(self, state, action):
        n_state = check_samples(state)
        n_action = check_samples(action)
        state = state.view(n_state, -1)
        action = action.view(n_action, -1)

        x1, x2 = self.model(state, action)
        x1 = x1[:, : self.n_heads, :]
        x2 = x2[:, : self.n_heads, :]
        return x1, x2

    def add_head(self, n_heads: int = 1):
        self.n_heads += n_heads
        assert (
            self.n_heads <= self.max_nheads
        ), f"exceed max num heads {self.max_nheads}"


if __name__ == "__main__":
    from torch.profiler import ProfilerActivity, profile, record_function

    obs_dim = 5
    featdim = 10
    act_dim = 2
    n_heads = 20
    hidden_dim = 64
    num_layers = 4

    resnet = False
    layernorm = False
    fta = False
    device = "cuda"

    times = 1000
    obs = 1000 * torch.rand(128, obs_dim).to(device)
    act = torch.rand(128, act_dim).to(device)

    sf1 = MultiheadSFNetwork(
        observation_dim=obs_dim,
        feature_dim=featdim,
        action_dim=act_dim,
        n_heads=n_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        resnet=resnet,
        layernorm=layernorm,
        fta=fta,
    ).to(device)
    sf1(obs, act)

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # ) as prof1:
    #     with record_function("model_inference"):
    #         for _ in range(times):
    #             sf1(obs, act)

    # print(prof1.key_averages().table(sort_by="cuda_time_total", row_limit=10))
