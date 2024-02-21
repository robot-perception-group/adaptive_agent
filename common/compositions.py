import functorch
import torch
from common.util import pile_sa_pairs
from torch.distributions import Normal

Epsilon = 1e-6


class Compositions:
    def __init__(
        self,
        agent_cfg,
        policy,
        sf,
        n_env,
        n_heads,
        action_dim,
        feature_dim,
        device,
    ) -> None:
        self.policy = policy
        self.sf = sf
        self.n_env = n_env
        self.n_heads = n_heads
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device

        self.norm_task_by_sf = agent_cfg["norm_task_by_sf"]
        self.use_auxiliary_task = agent_cfg["use_auxiliary_task"]
        self.n_auxTask = agent_cfg["n_auxTask"]

        self.record_impact = False
        self.impact_x_idx = []  # record primitive impact on x-axis
        self.policy_idx = []  # record primitive summon frequency

        self.mask_nullcomp = (
            torch.eye(self.n_heads).unsqueeze(dim=-1).bool().to(self.device)
        )

    def composition_methods(self, method="sfgpi"):
        if method == "dac" or method == "dacgpi":
            self.record_impact = True
        else:
            self.record_impact = False

        if method == "sfgpi":
            return self.sfgpi
        elif method == "msf":
            return self.msf
        elif method == "sfcpi":
            return self.sfcpi
        elif method == "dac":
            return self.dac
        elif method == "dacgpi":
            return self.dacgpi
        else:
            raise NotImplementedError

    def add_head(self, nheads):
        self.n_heads += nheads
        self.mask_nullcomp = (
            torch.eye(self.n_heads).unsqueeze(dim=-1).bool().to(self.device)
        )

    def act(self, s, w, id, mode="exploit", composition="sfgpi"):
        if composition == "null" or composition is None:
            return self.null_comp(s, id)

        return self.composition_methods(composition)(s, w, mode)

    def null_comp(self, s, id):
        acts, _, dist, _ = self.forward_policy(s)  # [N, H, A]  <-- [N, S]
        act_dim = acts.shape[2]

        a = torch.masked_select(acts, self.mask_nullcomp[id]).view(
            -1, act_dim
        )  # [N, A]
        return a, dist

    def sfgpi(self, s, w, mode):
        if mode == "explore":
            acts, _, dist, _ = self.forward_policy(s)  # [N, Ha, A]  <-- [N, S]
        elif mode == "exploit":
            _, _, dist, acts = self.forward_policy(s)  # [N, Ha, A]  <-- [N, S]

        qs = self.gpe(s, acts, w)  # [N, Ha, Hsf] <-- [N, S], [N, Ha, A], [N, F]
        a = self.gpi(acts, qs)  # [N, A] <-- [N, Ha, A], [N, Ha, Hsf]
        return a, dist

    def msf(self, s, w, mode):
        means, log_stds = self.forward_policy_distribution(s)  # [N, H, A]
        composed_mean, composed_std = self.cpi(means, log_stds, w, rule="mcp")
        if mode == "explore":
            a = torch.tanh(Normal(composed_mean, composed_std).rsample())
        elif mode == "exploit":
            a = composed_mean
        return a

    def sfcpi(self, s, w, mode):
        means, log_stds = self.forward_policy_distribution(s)  # [N, H, A]
        # [N, Ha, Hsf] <-- [N, S], [N, H, A], [N, F]
        qs = self.gpe(s, means, w)
        qs = qs.mean(2)  # [N, Ha]
        composed_mean, composed_std = self.cpi(means, log_stds, qs, rule="mcp")
        if mode == "explore":
            a = torch.tanh(Normal(composed_mean, composed_std).rsample())
        elif mode == "exploit":
            a = composed_mean
        return a

    def dac(self, s, w, mode):
        means, log_stds = self.forward_policy_distribution(s)  # [N, H, A]
        kappa = self.cpe(s, means, w)
        composed_mean, composed_std = self.cpi(means, log_stds, kappa, rule="mca")
        if mode == "explore":
            a = torch.tanh(Normal(composed_mean, composed_std).rsample())
        elif mode == "exploit":
            a = composed_mean
        return a

    def dacgpi(self, s, w, mode):
        if mode == "explore":
            a, _, dist, _ = self.forward_policy(s)  # [N, H, A]  <-- [N, S]
        elif mode == "exploit":
            _, _, dist, a = self.forward_policy(s)  # [N, H, A]  <-- [N, S]

        kappa = self.cpe(s, a, w)
        a = self.gpi(a, kappa, rule="k")
        return a, dist

    def gpe(self, s, a, w):
        # [N, Ha, Hsf, F] <-- [N, S], [N, Ha, A]
        curr_sf = self.forward_sf(s, a, self.sf)

        if self.norm_task_by_sf:
            w /= curr_sf.mean([0, 1, 2]).abs()  # normalized by SF scale
            w /= (w.norm(1, 1, keepdim=True)+1e-6)  # [N, Ha], H=F

        # [N,Ha,Hsf]<--[N,Ha,Hsf,F],[N,F]
        qs = torch.einsum("ijkl,il->ijk", curr_sf, w)

        return qs  # [N,Ha,Hsf]

    def gpi(self, acts, value, rule="q"):
        if rule == "q":
            value_flat = value.flatten(1)  # [N, Ha*Hsf] <-- [N, Ha, Hsf]
            idx_max = value_flat.argmax(1)  # [N]
            idx = torch.div(idx_max, self.n_heads, rounding_mode="floor")  # [N]
            idx = idx[:, None].repeat(1, self.action_dim).unsqueeze(1)  # [N,1,A]<-[N]
        elif rule == "k":
            idx = value.argmax(1).unsqueeze(1)  # [N, 1, A] <-- [N, H, A]

        a = torch.gather(acts, 1, idx).squeeze(1)  # [N, A] <-- [N, Ha, A]
        # record policy freqeuncy
        self.policy_idx.extend(idx.reshape(-1).cpu().numpy())
        return a  # [N, A]

    def cpe(self, s, a, w):
        curr_sf = self.forward_sf(s, a, self.sf)  # [N,Ha,Hsf,F]<--[N,S],[N,Ha,A]
        impact = self.calc_impact(s, a)  # [N,F,A]<--[N,Ha,Hsf,F],[N,Ha,A]
        kappa = self.calc_advantage(curr_sf)  # [N,Hsf,F] <-- [N, Ha, Hsf, F]
        kappa = torch.relu(kappa)  # filterout advantage below 0
        # [N,H,A]<--[N,Hsf,F], [N,F], [N,F,A]
        kappa = torch.einsum("ijk,ik,ikl->ijl", kappa, w, impact)
        return kappa

    def cpi(self, means, log_stds, gating, rule="mcp", k=0.9):
        # select top k% value
        gating -= torch.amin(gating, dim=1, keepdim=True)
        gating_max = torch.amax(gating, dim=1, keepdim=True).repeat_interleave(
            gating.shape[1], 1
        )
        gating = torch.where(gating < gating_max * k, 0, gating)
        gating /= gating.norm(1, 1, keepdim=True)  # [N, Ha], H=F

        if rule == "mcp":
            # [N, H, A] <-- [N,F], [N,H,A], F=H
            w_div_std = torch.einsum("ij, ijk->ijk", gating, (-log_stds).exp())
        elif rule == "mca":
            # [N, H, A] <-- [N,F,H], [N,H,A], F=H
            w_div_std = torch.einsum("ijk, ijk->ijk", gating, (-log_stds).exp())

        composed_std = 1 / (w_div_std.sum(1) + Epsilon)  # [N,A]
        # [N,A]<--[N,A]*[N,A]<-- [N,H,A], [N,H,A]
        composed_mean = composed_std * torch.einsum("ijk,ijk->ik", means, w_div_std)
        return composed_mean, composed_std

    def forward_policy(self, s):
        acts, entropy, dist, mean = self.policy.sample(s)  # [N, H, A]
        return acts, entropy, dist, mean

    def forward_policy_distribution(self, s):
        means, log_stds = self.policy.get_mean_std(s)  # [N, H, A]
        return means, log_stds

    def forward_sf(self, s, a, sf_model):
        # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]
        s_tiled, a_tiled = pile_sa_pairs(s, a)

        # [NHa, Hsf, F] <-- [NHa, S], [NHa, A]
        sf1, sf2 = sf_model(s_tiled, a_tiled)
        sf = torch.min(sf1, sf2)

        if self.use_auxiliary_task:
            sf = sf[:, : -self.n_auxTask]

        # [N, Ha, Hsf, F]
        sf = sf.view(-1, self.n_heads, self.n_heads, self.feature_dim)
        return sf

    def calc_advantage(self, value):  # [N,Ha,Hsf,F]
        adv = value.mean(1, keepdim=True) - value.mean((1, 2), keepdim=True)
        return adv.squeeze(1)

    def calc_impact(self, s, a):
        # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]
        s, a = pile_sa_pairs(s, a)

        self.sf = self.sf.eval()

        def func(s, a):
            if self.use_auxiliary_task:
                return torch.min(*self.sf(s, a))[:, : -self.n_auxTask]  # [NHa, Hsf, F]
            else:
                return torch.min(*self.sf(s, a))  # [NHa, Hsf, F]

        j = functorch.vmap(functorch.jacrev(func, argnums=1))(s, a)  # [NHa,Hsf,F,A]
        j = j.view(-1, self.n_heads, self.n_heads, self.feature_dim, self.action_dim)
        cur_impact = j.mean((1, 2)).abs()  # [N,F,A]<-[N,Ha,Hsf,F,A]

        self.sf = self.sf.train()

        impact = (self.prev_impact + cur_impact) / 2
        self.prev_impact = impact

        # record primitive impact
        idx = impact.argmax(1)
        self.impact_x_idx.extend(idx[:, 0].reshape(-1).cpu().numpy())
        return impact

    def reset(self):
        self.prev_impact = torch.zeros(
            (self.n_env, self.feature_dim, self.action_dim), device=self.device
        )
        self.impact_x_idx = []  # record primitive impact on x-axis
        self.policy_idx = []  # record primitive summon frequency
