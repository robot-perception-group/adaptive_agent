import copy
from pathlib import Path

import isaacgym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal

from common.agent import MultitaskAgent
from common.compositions import Compositions
from common.policy import MultiheadGaussianPolicy
from common.util import (
    check_act,
    check_obs,
    grad_false,
    grad_true,
    hard_update,
    np2ts,
    pile_sa_pairs,
    soft_update,
    update_params,
    check_samples,
)
from common.value import MultiheadSFNetwork
from common.vec_buffer import FrameStackedReplayBuffer
from torch.optim import Adam


class CompAgent(MultitaskAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.load_model = self.agent_cfg.get("load_model", False)
        self.model_path = self.agent_cfg.get("model_path", None)

        self.framestacked_replay = self.buffer_cfg["framestacked_replay"]
        self.stack_size = self.buffer_cfg["stack_size"]
        assert (
            self.framestacked_replay == True
        ), "This agent only support framestacked replay"

        self.lr = self.agent_cfg["lr"]
        self.policy_lr = self.agent_cfg["policy_lr"]
        self.adaptor_lr = self.agent_cfg["adaptor_lr"]
        self.lr_schedule = self.agent_cfg["lr_schedule"]
        self.sf_net_kwargs = self.agent_cfg["sf_net_kwargs"]
        self.policy_net_kwargs = self.agent_cfg["policy_net_kwargs"]
        self.gamma = self.agent_cfg["gamma"]
        self.tau = self.agent_cfg["tau"]

        self.td_target_update_interval = int(
            self.agent_cfg["td_target_update_interval"]
        )
        self.updates_per_step = self.agent_cfg["updates_per_step"]
        self.grad_clip = self.agent_cfg["grad_clip"]
        self.entropy_tuning = self.agent_cfg["entropy_tuning"]

        self.use_auxiliary_task = self.agent_cfg["use_auxiliary_task"]
        self.aux_coeff = self.agent_cfg.get("aux_coeff", 0)
        self.use_continuity_loss = self.agent_cfg["use_continuity_loss"]
        self.continuity_coeff = self.agent_cfg["continuity_coeff"]
        self.use_imitation_loss = self.agent_cfg["use_imitation_loss"]
        self.imitation_coeff = self.agent_cfg["imitation_coeff"]
        self.use_kl_loss = self.agent_cfg["use_kl_loss"]
        self.kl_coeff = self.agent_cfg["kl_coeff"]
        self.use_collective_learning = self.agent_cfg.get(
            "use_collective_learning", False
        )

        self.curriculum = (
            self.agent_cfg["curriculum_learning"]
            if self.env_cfg["task"]["domain_rand"]
            else False
        )

        self.explore_method = self.agent_cfg.get("explore_method", "null")
        self.exploit_method = self.agent_cfg.get("exploit_method", "sfgpi")

        self.wandb_verbose = self.agent_cfg.get("wandb_verbose", False)

        self.env_latent_dim = self.env.num_latent  # E
        self.env_expert_dim = self.env.num_expert  # Ex
        self.observation_dim = (
            self.observation_dim - self.env_latent_dim - self.env_expert_dim
        )  # S = O-E-Ex
        self.env_latent_idx = self.observation_dim + 1
        self.env_expert_idx = self.env_latent_idx + self.env_latent_dim
        self.n_expert = self.env_expert_dim // self.action_dim

        # define primitive tasks
        self.w_primitive = self.task.Train.taskSet
        self.n_heads = self.w_primitive.shape[0]
        assert (
            self.n_heads <= self.sf_net_kwargs["max_nheads"]
        ), f"number of task {self.n_heads} exceed the maximum"

        self.replay_buffer = FrameStackedReplayBuffer(
            obs_shape=self.observation_shape,
            action_shape=self.action_shape,
            feature_shape=self.feature_shape,
            n_heads=self.n_heads,
            device=self.device,
            **self.buffer_cfg,
        )

        # define models
        if self.use_auxiliary_task:
            self.n_auxTask = 1
            self.n_sfhead = self.n_heads + self.n_auxTask
            self.aux_loss = nn.MSELoss()
        else:
            self.n_sfhead = self.n_heads

        self.sf = MultiheadSFNetwork(
            observation_dim=self.observation_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            n_heads=self.n_sfhead,
            **self.sf_net_kwargs,
        ).to(self.device)

        self.sf_target = MultiheadSFNetwork(
            observation_dim=self.observation_dim,
            feature_dim=self.feature_dim,
            action_dim=self.action_dim,
            n_heads=self.n_sfhead,
            **self.sf_net_kwargs,
        ).to(self.device)

        hard_update(self.sf_target, self.sf)
        grad_false(self.sf_target)
        self.sf_loss = nn.MSELoss()

        self.policy = MultiheadGaussianPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            **self.policy_net_kwargs,
        ).to(self.device)

        if self.load_model and self.model_path is not None:
            self.model_path = self.env_cfg["log_path"] + self.model_path
            print("load model:", self.model_path)
            self.load_torch_model(self.model_path)

        params = [
            {"params": self.sf.parameters()},
        ]

        self.sf_optimizer = Adam(
            params,
            lr=self.lr,
            betas=[0.9, 0.999],
        )
        self.policy_optimizer = Adam(
            self.policy.parameters(),
            lr=self.policy_lr,
            betas=[0.9, 0.999],
        )

        if self.lr_schedule:
            self.lrScheduler_sf = torch.optim.lr_scheduler.LinearLR(
                self.sf_optimizer,
                start_factor=1,
                end_factor=0.2,
                total_iters=self.episode_max_step * self.total_episodes,
            )
            self.lrScheduler_policy = torch.optim.lr_scheduler.LinearLR(
                self.policy_optimizer,
                start_factor=1,
                end_factor=0.2,
                total_iters=self.episode_max_step * self.total_episodes,
            )

        self.comp = Compositions(
            self.agent_cfg,
            self.policy,
            self.sf,
            self.n_env,
            self.n_heads,
            self.action_dim,
            self.feature_dim,
            self.device,
        )

        self._create_sf_mask()

        if self.entropy_tuning:
            self._create_entropy_tuner()
        else:
            self.alpha = torch.tensor(self.agent_cfg["alpha"], device=self.device)

        # init params
        self.learn_steps = 0

        if self.use_kl_loss:
            self.dist = Normal(
                torch.zeros(
                    self.mini_batch_size,
                    self.n_heads,
                    self.action_dim,
                    device=self.device,
                ),
                torch.ones(
                    self.mini_batch_size,
                    self.n_heads,
                    self.action_dim,
                    device=self.device,
                ),
            )

        if self.curriculum:
            self.total_curistages = 20
            self.curri_stage = 0
            self.curri_epi = np.linspace(
                0, self.total_episodes - 1, num=self.total_curistages
            )
            self.lat_ra = self.env_cfg["task"]["range_a"]
            self.lat_rb = self.env_cfg["task"]["range_b"]

        self.idx_trig = 7

    def run(self):
        grad_true(self.sf)
        grad_true(self.sf_target)
        grad_true(self.policy)
        self.train_phase(self.episodes + self.total_episodes, self.curriculum)

        print(f"============= finish =============")

    def update_curri_stage(self):
        self.curri_stage += 1

        step_size_a = (
            1
            / self.total_curistages
            * (self.lat_ra[1] - self.lat_ra[0])
            * self.curri_stage
        )
        step_size_b = (
            1
            / self.total_curistages
            * (self.lat_rb[1] - self.lat_rb[0])
            * self.curri_stage
        )

        range_a = [1 - step_size_a, 1 + step_size_a]
        range_b = [1 - step_size_b, 1 + step_size_b]

        self.env.set_latent_range(range_a, range_b)

    def train_phase(self, episodes, curriculum=False):
        print(f"============= start =============")
        while True:
            self.train_episode()

            if self.eval and (self.episodes % self.eval_interval == 0):
                self.evaluate()

                if self.save_model:
                    self.save_torch_model()

            if self.episodes > episodes - 1:
                break

            if curriculum and self.episodes >= int(self.curri_epi[self.curri_stage]):
                self.update_curri_stage()

    def train_episode(self, gui_app=None, gui_rew=None):
        self.episodes += 1
        episode_r = episode_steps = trigger_wp = hover_time = 0
        done = False

        print("episode = ", self.episodes)
        self.task.rand_task(self.episodes)

        s = self.reset_env()
        for _ in range(self.episode_max_step):
            episodeLen = self.env.progress_buf.clone()

            s_next, r, done = self.step(episode_steps, s)

            s = s_next
            self.steps += self.n_env
            trigger_wp += s[:, self.idx_trig].sum()
            hover_time += (torch.norm(s[:, 8 : 8 + 3], dim=1) < 7).sum()
            episode_steps += 1
            episode_r += r

            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.size()[0]:
                self.game_rewards.update(episode_r[done_ids])
                self.game_lengths.update(episodeLen[done_ids])

            # call gui update loop
            if gui_app:
                gui_app.update_idletasks()
                gui_app.update()
                self.avgStepRew.update(r)
                gui_rew.set(self.avgStepRew.get_mean())

            if episode_steps >= self.episode_max_step:
                break

        task_return = self.task.trainTaskR(episode_r)

        if self.adaptive_task:
            self.task.adapt_task()

        metrics = trigger_wp + hover_time / 100
        wandb.log(
            {
                f"reward/train": self.game_rewards.get_mean(),
                f"reward/episode_length": self.game_lengths.get_mean(),
                f"reward/ntriggers": trigger_wp.detach().item(),
                f"reward/hovertime": hover_time.detach().item(),
                f"reward/metrics": metrics.detach().item(),
            }
        )
        if self.curriculum:
            wandb.log({"reward/curriculum_stage": self.curri_stage})

        if self.wandb_verbose:
            task_return = task_return.detach().tolist()
            for i in range(len(task_return)):
                wandb.log(
                    {
                        f"reward/task_return{i}": task_return[i],
                    }
                )

        return episode_r, episode_steps, {}

    def evaluate(self):
        episodes = int(self.eval_episodes)
        if episodes == 0:
            return

        print(
            f"===== evaluate at episode: {self.episodes} for {self.episode_max_step} steps ===="
        )

        returns = torch.zeros((episodes,), dtype=torch.float32)
        for i in range(episodes):
            episode_r = 0.0
            trigger_wp = 0
            hover_time = 0

            s = self.reset_env()
            for _ in range(self.episode_max_step):
                a = self.act(s, self.task.Eval, "exploit")
                self.env.step(a)
                s_next = self.env.obs_buf.clone()
                self.env.reset()

                r = self.calc_reward(s_next, self.task.Eval.W)

                s = s_next
                episode_r += r
                trigger_wp += s[:, self.idx_trig].sum()
                hover_time += (torch.norm(s[:, 8 : 8 + 3], dim=1) < 7).sum()

            returns[i] = torch.mean(episode_r).item()

        metrics = trigger_wp + hover_time / 100

        print(
            f"eval episode {self.episodes}: ntrigger {trigger_wp}, hover_time {hover_time},  metrics {metrics}"
        )
        print(f"===== finish evaluate ====")

        wandb.log(
            {
                f"reward/eval": torch.mean(returns).item(),
                f"reward/eval_ntriggers": trigger_wp.detach().item(),
                f"reward/eval_hovertime": hover_time.detach().item(),
                f"reward/eval_metrics": metrics.detach().item(),
                f"state/eval_policy_idx": wandb.Histogram(self.comp.policy_idx),
            }
        )
        if self.wandb_verbose:
            task_return = self.task.evalTaskR(episode_r)
            task_return = task_return.detach().tolist()
            for i in range(len(task_return)):
                wandb.log(
                    {
                        f"reward/task_return{i}": task_return[i],
                    }
                )

        return returns, {}

    def step(self, episode_steps, s):
        assert not torch.isnan(
            s
        ).any(), f"detect anomaly state {(torch.isnan(s)==True).nonzero()}"

        a = self.act(s, self.task.Train)
        assert not torch.isnan(
            a
        ).any(), f"detect anomaly action {(torch.isnan(a)==True).nonzero()}"

        self.env.step(a)
        done = self.env.reset_buf.clone()
        s_next = self.env.obs_buf.clone()
        self.env.reset()

        assert not torch.isnan(
            s_next
        ).any(), f"detect anomaly state {(torch.isnan(s_next)==True).nonzero()}"

        r = self.calc_reward(s_next, self.task.Train.W)

        masked_done = False if episode_steps >= self.episode_max_step else done

        self.save_to_buffer(s, a, r, s_next, done, masked_done)

        if self.is_update():
            for _ in range(self.updates_per_step):
                self.learn()

        return s_next, r, done

    def act(self, o, task, mode="explore"):
        try:
            o = check_obs(
                o, self.observation_dim + self.env_latent_dim + self.env_expert_dim
            )
        except:
            o = check_obs(o, self.observation_dim)

        # [N, A], [N, H, A] <-- [N, S+E]
        a = self._act(o, task, mode)

        a = check_act(a, self.action_dim)
        return a

    def _act(self, o, task, mode):
        with torch.no_grad():
            if (
                (self.steps <= self.min_n_experience)
                and mode == "explore"
                and not self.load_model
            ):
                a = (
                    2 * torch.rand((self.n_env, self.env.num_act), device=self.device)
                    - 1
                )

            w = copy.copy(np2ts(task.W))
            id = copy.copy(np2ts(task.id))

            if mode == "explore":
                a = self.explore(o, w, id)
            elif mode == "exploit":
                a = self.exploit(o, w, id)
        return a

    def explore(self, o, w, id):
        s, _, _ = self.parse_state(o)
        a, _ = self.comp.act(s, w, id, "explore", self.explore_method)
        return a

    def exploit(self, o, w, id):
        s, _, _ = self.parse_state(o)
        a, _ = self.comp.act(s, w, id, "exploit", self.exploit_method)
        return a

    def parse_state(self, s):
        if s.shape[1] >= self.env_latent_idx:
            # parse state and env_latent if env_latent is included in the observation
            # [N, S], [N,E], [N,Ex]
            s, e, ex = (
                s[:, : self.env_latent_idx - 1],
                s[:, self.env_latent_idx - 1 : self.env_expert_idx - 1],
                s[:, self.env_expert_idx - 1 :],
            )
        else:
            e = None
            ex = None

        return s, e, ex

    def reset_env(self):
        self.comp.reset()
        return super().reset_env()

    def learn(self):
        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            soft_update(self.sf_target, self.sf, self.tau)

        batch = self.replay_buffer.sample(self.mini_batch_size)

        sf_loss, info_sf = self.update_sf(batch)
        policy_loss, entropies, info_pi = self.update_policy(batch)

        if self.lr_schedule:
            self.lrScheduler_sf.step()
            self.lrScheduler_policy.step()

        if self.entropy_tuning:
            entropy_loss = self._calc_entropy_loss(entropies)
            update_params(self.alpha_optimizer, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
            # self.alpha = torch.clip(self.alpha, 0, 2)

        # start logging
        if self.learn_steps % self.log_interval == 0:
            metrics = {
                "loss/SF": sf_loss,
                "loss/policy": policy_loss,
                "loss/action_norm": info_pi["action_norm"],
                "state/mean_SF1": info_sf["curr_sf1"],
                "state/mean_SF2": info_sf["curr_sf2"],
                "state/target_sf": info_sf["target_sf"],
                "state/entropy": entropies.detach().mean().item(),
                "state/policy_idx": wandb.Histogram(self.comp.policy_idx),
                "state/lr_sf": self.sf_optimizer.param_groups[0]["lr"],
                "state/lr_policy": self.policy_optimizer.param_groups[0]["lr"],
            }
            if self.use_auxiliary_task:
                metrics.update(
                    {
                        "loss/auxiliary_loss": info_sf["auxiliary_loss"],
                    }
                )
            if self.use_imitation_loss:
                metrics.update(
                    {
                        "loss/imitation_loss": info_pi["imitation_loss"],
                        "loss/imitation_loss0": info_pi["imitation_loss0"],
                        "loss/imitation_loss1": info_pi["imitation_loss1"],
                        "loss/imitation_loss2": info_pi["imitation_loss2"],
                        "loss/imitation_loss3": info_pi["imitation_loss3"],
                    }
                )
            if self.use_kl_loss:
                metrics.update(
                    {
                        "loss/approx_kl": info_pi["approx_kl"],
                    }
                )
            if self.use_continuity_loss:
                metrics.update(
                    {
                        "loss/action_continuity": -info_pi["continuity_loss"],
                    }
                )

            if self.comp.record_impact:
                metrics.update(
                    {
                        "state/impact_x_idx": wandb.Histogram(self.comp.impact_x_idx),
                    }
                )

            if self.entropy_tuning:
                metrics.update(
                    {
                        "loss/alpha": entropy_loss.detach().item(),
                        "state/alpha": self.alpha.mean().detach().item(),
                    }
                )
            wandb.log(metrics)

    def update_sf(self, batch):
        (s, a, s_next, dones) = (
            batch["obs"],
            batch["action"],
            batch["next_obs"],
            batch["done"],
        )
        s, _, _ = self.parse_state(s)
        s_next, _, _ = self.parse_state(s_next)

        f = self.feature.extract(s).to(torch.float32)

        # compute sf loss
        curr_sf1, curr_sf2 = self.sf(s, a)  # [N, Hsf, F] <--[N, S+Z], [N,A]
        target_sf = self._calc_target_sf(f, s_next, dones)  # [N, Ha, F]

        if self.use_auxiliary_task:
            f_next = self.feature.extract(s_next).to(torch.float32)
            f_next_pred1 = curr_sf1[:, -1]
            f_next_pred2 = curr_sf2[:, -1]

            auxiliary_loss = self.aux_loss(f_next_pred1, f_next) + self.aux_loss(
                f_next_pred2, f_next
            )

            curr_sf1 = curr_sf1[:, : -self.n_auxTask]  # [N, Ha, F]
            curr_sf2 = curr_sf2[:, : -self.n_auxTask]  # [N, Ha, F]

        sf_loss = self.sf_loss(curr_sf1, target_sf) + self.sf_loss(curr_sf2, target_sf)

        self.sf_optimizer.zero_grad()
        if self.use_auxiliary_task:
            auxiliary_loss.backward(retain_graph=True)
        sf_loss.backward()
        self.sf_optimizer.step()

        # log to monitor training.
        info = {}
        info["curr_sf1"] = curr_sf1.detach().mean().item()
        info["curr_sf2"] = curr_sf2.detach().mean().item()
        info["target_sf"] = target_sf.detach().mean().item()
        if self.use_auxiliary_task:
            info["auxiliary_loss"] = auxiliary_loss.detach().item()

        return sf_loss.detach().item(), info

    def update_policy(self, batch):
        s_stack, a_stack = batch["stacked_obs"], batch["stacked_act"]
        s, e, ex = self.parse_state(s_stack[:, :, 0])  # [N, S], [N, E], [N, Ex]

        # [N,H,A], [N, H, 1] <-- [N,S+Z]
        a_heads, entropies, dist, _ = self.policy.sample(s)

        if self.use_collective_learning:
            qs = self._collective_learning(
                s, a_heads, self.w_primitive
            )  # [N, H, 1] <-- [N, S], [N, H, A], [H, F]
        else:
            qs = self._calc_qs_from_sf(s, a_heads)  # [N,H]<--[N, S+Z], [N, H, A]
            qs = qs.unsqueeze(2)  # [N,H,1]

        policy_loss = -qs - self.alpha * entropies  # [N,H,1]
        policy_loss_record = policy_loss.mean().detach().item()

        # monitor
        info = {}
        action_norm = torch.norm(a_heads, p=1, dim=2, keepdim=True) / self.action_dim
        info["action_norm"] = action_norm.mean().detach().item()

        if self.use_continuity_loss:  # encourage continuous action
            prev_a = a_stack[..., 1]  # [N, A] <-- [N, A, K]

            # [N,H,1] <-- [N,1,A] - [N,H,A]
            continuity_loss = torch.norm(
                prev_a[:, None, :] - a_heads, dim=2, keepdim=True
            )
            policy_loss = (
                policy_loss + (1 - self.alpha) * self.continuity_coeff * continuity_loss
            )
            info["continuity_loss"] = continuity_loss.mean().detach().item()

        if self.use_imitation_loss:
            imi_loss = torch.zeros_like(policy_loss)  # [N, H, 1]
            for i in range(self.n_expert):
                controller = ex[:, self.action_dim * i : self.action_dim * (i + 1)]
                imi_loss[:, i] = torch.norm(
                    controller - a_heads[:, i], dim=1, keepdim=True
                )
                info[f"imitation_loss{i}"] = imi_loss[:, i].mean().detach().item()
            info["imitation_loss"] = imi_loss.mean().detach().item()

            policy_loss = policy_loss + self.imitation_coeff * imi_loss

        if self.use_kl_loss:
            with torch.no_grad():
                prev_logp = self.dist.log_prob(a_heads)
                self.dist = dist

            logratio = dist.log_prob(a_heads) - prev_logp  # [N, H, A]
            logratio = torch.clamp(logratio, -0.22, 0.18)
            approx_kl = ((logratio.exp() - 1) - logratio).mean(dim=2, keepdim=True)

            policy_loss = policy_loss + self.kl_coeff * approx_kl

            info["approx_kl"] = approx_kl.mean().detach().item()

        policy_loss = torch.mean(policy_loss)
        update_params(self.policy_optimizer, self.policy, policy_loss, self.grad_clip)

        return policy_loss_record, entropies, info

    def _collective_learning(self, s, a, w):
        # [N, Ha, Hsf, F] <-- [N, S], [N, Ha, A]
        sf = self.comp.forward_sf(s, a, self.sf)

        # [N, Ha, Hsf]<--[N, Ha, Hsf, F], [Ha, F]
        qs = torch.einsum("ijkl,jl->ijk", sf, w)

        # [N, Ha, 1] <-- [N, Ha, Hsf]
        # qs = torch.logsumexp(qs, dim=2, keepdim=True) - torch.log(torch.tensor(self.n_heads, dtype=torch.float32))
        qs = torch.max(qs, dim=2, keepdim=True)[0]
        return qs

    def _calc_entropy_loss(self, entropy):
        # [N, H, 1] <--  [N, H, 1], [N,1,1]
        loss = self.log_alpha * (self.target_entropy - entropy).detach()
        entropy_loss = -torch.mean(loss)
        return entropy_loss

    def _calc_qs_from_sf(self, s, a):
        # [NHa, S], [NHa, A] <-- [N, S], [N, Ha, A]
        s_tiled, a_tiled = pile_sa_pairs(s, a)

        # [NHa, Hsf, F] <-- [NHa, S], [NHa, A]
        curr_sf1, curr_sf2 = self.sf(s_tiled, a_tiled)

        # [N, Ha, F] <-- [NHa, Hsf, F]
        curr_sf = self._process_sfs(curr_sf1, curr_sf2)

        # [N,H]<-- [N,H,F]*[H,F]
        qs = torch.einsum("ijk,jk->ij", curr_sf, self.w_primitive)

        return qs

    def _calc_target_sf(self, f, s, dones):
        _, _, _, a = self.policy.sample(s)  # [N, H, A] <-- [N, S+Z]

        with torch.no_grad():
            # [NHa, S+Z], [NHa, A] <-- [N, S+Z], [N, Ha, A]
            s_tiled, a_tiled = pile_sa_pairs(s, a)

            # [NHa, Hsf, F] <-- [NHa, S+Z], [NHa, A]
            next_sf1, next_sf2 = self.sf_target(s_tiled, a_tiled)

            # [N, Ha, F] <-- [NHa, Hsf, F]
            next_sf = self._process_sfs(next_sf1, next_sf2)

        # [N,Ha,F] <-- [N,F]
        f = torch.tile(f[:, None, :], (self.n_heads, 1))

        # [N, Ha, F] <-- [N, Ha, F] + [N, Ha, F]
        target_sf = f + torch.einsum("ijk,il->ijk", next_sf, (~dones) * self.gamma)

        return target_sf  # [N, Ha, F]

    def _process_sfs(self, sf1, sf2):
        sf = torch.min(sf1, sf2)  # [NHa, Hsf, F]

        if self.use_auxiliary_task:
            sf = sf[:, : -self.n_auxTask]  # [NHa, Hsf-k, F]

        # [NHa, F] <-- [NHa, Hsf, F], [NHa, Hsf, F]
        sf = torch.masked_select(sf, self.mask)

        # [N, Ha, F] <-- [NHa, F]
        sf = sf.view(self.mini_batch_size, self.n_heads, self.feature_dim)
        return sf

    def _create_entropy_tuner(self):
        self.alpha_lr = self.agent_cfg["alpha_lr"]
        target_entropy_scale = self.agent_cfg.get("target_entropy_scale", 1.0)

        self.target_entropy = -target_entropy_scale * torch.prod(
            torch.tensor(self.action_shape, device=self.device)
        ).tile(self.n_heads).unsqueeze(
            1
        )  # -|A|, [H,1]

        # optimize log(alpha), instead of alpha
        if hasattr(self, "log_alpha"):
            self.log_alpha = torch.concat(
                [self.log_alpha.data, torch.zeros([1, 1])]
            ).to(self.device)
            self.log_alpha.requires_grad = True
        elif self.load_model:  # when load model, reduce entropy bonus
            self.log_alpha = torch.zeros((self.n_heads, 1), device=self.device) - 2.0
            self.log_alpha.requires_grad = True
        else:
            self.log_alpha = torch.zeros(
                (self.n_heads, 1), requires_grad=True, device=self.device
            )

        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.alpha_lr)

    def _create_sf_mask(self):
        # masks used for vectorizing SFs head selction
        # [H, H, 1]
        self.mask = torch.eye(self.n_heads, device=self.device).unsqueeze(dim=-1)
        # [NH, H, F]
        self.mask = self.mask.repeat(self.mini_batch_size, 1, self.feature_dim)
        self.mask = self.mask.bool()

    def add_task(self, w):
        w = w.view(-1, self.feature_dim)
        ntask = w.shape[0]

        # update submodules
        self.task.add_task(w)
        self.sf.add_head(ntask)
        self.sf_target.add_head(ntask)
        self.policy.add_head(ntask)
        self.comp.add_head(ntask)

        # update module
        self.n_heads += ntask
        self.w_primitive = self.task.Train.taskSet
        self._create_sf_mask()

        if self.entropy_tuning:
            self._create_entropy_tuner()

    def setup_replaybuffer(self):
        super().setup_replaybuffer()

        if self.buffer_cfg["framestacked_replay"]:
            self.replay_buffer = FrameStackedReplayBuffer(
                obs_shape=self.observation_shape,
                action_shape=self.action_shape,
                device=self.device,
                n_heads=self.n_heads,
                **self.buffer_cfg,
            )

    def save_torch_model(self):
        path = self.log_path + f"model{self.episodes}/"

        print("saving model at ", path)
        Path(path).mkdir(parents=True, exist_ok=True)
        self.policy.save(path + "policy")
        self.sf.save(path + "sf")

    def load_torch_model(self, path):
        self.policy.load(path + "/policy")
        self.sf.load(path + "/sf")

        hard_update(self.sf_target, self.sf)
        grad_false(self.sf_target)
