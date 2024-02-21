import copy
import datetime
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import torch
from common.util import AverageMeter, check_act, check_obs, dump_cfg, np2ts
from common.vec_buffer import (
    VectorizedReplayBuffer,
    FrameStackedReplayBuffer,
    VecPrioritizedReplayBuffer,
)
from env.wrapper.multiTask import multitaskenv_constructor

import wandb

warnings.simplefilter("once", UserWarning)
exp_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class AbstractAgent(ABC):
    @abstractmethod
    def act(self, s):
        pass

    @abstractmethod
    def step(self):
        pass


class IsaacAgent(AbstractAgent):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.env_cfg = cfg["env"]
        self.agent_cfg = cfg["agent"]
        self.buffer_cfg = cfg["buffer"]
        self.device = cfg["rl_device"]

        self.env, self.feature, self.task = multitaskenv_constructor(
            env_cfg=self.env_cfg, device=self.device
        )
        assert self.feature.dim == self.task.dim, "feature and task dimension mismatch"

        self.n_env = self.env_cfg["num_envs"]
        self.episode_max_step = self.env_cfg["episode_max_step"]
        self.total_episodes = int(self.env_cfg["total_episodes"])
        self.total_timesteps = self.n_env * self.episode_max_step * self.total_episodes

        self.log_interval = self.env_cfg["log_interval"]
        self.eval = self.env_cfg["eval"]
        self.eval_interval = self.env_cfg["eval_interval"]
        self.eval_episodes = self.env_cfg["eval_episodes"]
        self.save_model = self.env_cfg["save_model"]

        self.observation_dim = self.env.num_obs
        self.feature_dim = self.feature.dim
        self.action_dim = self.env.num_act
        self.observation_shape = [self.observation_dim]
        self.feature_shape = [self.feature_dim]
        self.action_shape = [self.action_dim]

        if self.buffer_cfg["prioritized_replay"]:
            self.replay_buffer = VecPrioritizedReplayBuffer(
                device=self.device,
                **self.buffer_cfg,
            )
        else:
            self.replay_buffer = VectorizedReplayBuffer(
                self.observation_shape,
                self.action_shape,
                device=self.device,
                **self.buffer_cfg,
            )
        self.mini_batch_size = int(self.buffer_cfg["mini_batch_size"])
        self.min_n_experience = int(self.buffer_cfg["min_n_experience"])

        self.gamma = int(self.agent_cfg["gamma"])
        self.updates_per_step = int(self.agent_cfg["updates_per_step"])
        self.reward_scale = int(self.agent_cfg["reward_scale"])

        if self.save_model:
            log_dir = (
                self.agent_cfg["name"]
                + "/"
                + self.env_cfg["env_name"]
                + "/"
                + exp_date
                + "/"
            )
            self.log_path = self.env_cfg["log_path"] + log_dir
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
            dcfg = DictConfig(cfg)
            dcfg = OmegaConf.to_object(dcfg)
            dump_cfg(self.log_path + "cfg", dcfg)

        self.steps = 0
        self.episodes = 0

        self.games_to_track = 100
        self.game_rewards = AverageMeter(1, self.games_to_track).to(self.device)
        self.game_lengths = AverageMeter(1, self.games_to_track).to(self.device)
        self.avgStepRew = AverageMeter(1, 20).to(self.device)

    def run(self):
        while True:
            self.train_episode()
            wandb.log({"reward/train": self.game_rewards.get_mean()})
            wandb.log({"reward/episode_length": self.game_lengths.get_mean()})

            if self.eval and (self.episodes % self.eval_interval == 0):
                returns = self.evaluate()
                wandb.log({"reward/eval": torch.mean(returns).item()})

                if self.save_model:
                    self.save_torch_model()

            if self.episodes >= self.total_episodes:
                break

    def train_episode(self, gui_app=None, gui_rew=None):
        self.episodes += 1
        episode_r = episode_steps = 0
        done = False

        print("episode = ", self.episodes)
        self.task.rand_task(self.episodes)

        s = self.reset_env()
        for _ in range(self.episode_max_step):
            episodeLen = self.env.progress_buf.clone()

            s_next, r, done = self.step(episode_steps, s)

            s = s_next
            self.steps += self.n_env
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

        return episode_r, episode_steps, {}

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

    def is_update(self):
        return (
            len(self.replay_buffer) > self.mini_batch_size
            and self.steps >= self.min_n_experience
        )

    def reset_env(self):
        s = self.env.obs_buf.clone()
        if s is None:
            s = torch.zeros((self.n_env, self.env.num_obs))

        return s

    def save_to_buffer(self, s, a, r, s_next, done, masked_done, weights=None):
        r = r[:, None] * self.reward_scale
        done = done[:, None]
        masked_done = masked_done[:, None]

        self.replay_buffer.add(s, a, r, s_next, masked_done, weights)

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

            s = self.reset_env()
            for _ in range(self.episode_max_step):
                a = self.act(s, self.task.Eval, "exploit")
                self.env.step(a)
                s_next = self.env.obs_buf.clone()
                self.env.reset()

                r = self.calc_reward(s_next, self.task.Eval.W)

                s = s_next
                episode_r += r

            returns[i] = torch.mean(episode_r).item()

        print(f"===== finish evaluate ====")
        return returns, {"episode_r": episode_r}

    def act(self, s, task, mode="explore"):
        s = check_obs(s, self.observation_dim)

        a = self._act(s, task, mode)

        a = check_act(a, self.action_dim)
        return a

    def _act(self, s, task, mode):
        with torch.no_grad():
            if (self.steps <= self.min_n_experience) and mode == "explore":
                a = (
                    2 * torch.rand((self.n_env, self.env.num_act), device=self.device)
                    - 1
                )

            w = copy.copy(np2ts(task.W))

            if mode == "explore":
                a = self.explore(s, w)
            elif mode == "exploit":
                a = self.exploit(s, w)
        return a

    def calc_reward(self, s, w):
        f = self.feature.extract(s)
        r = torch.sum(w * f, 1)
        return r

    def explore(self):
        raise NotImplementedError

    def exploit(self):
        raise NotImplementedError

    def learn(self):
        pass

    def save_torch_model(self):
        raise NotImplementedError

    def load_torch_model(self):
        raise NotImplementedError


class MultitaskAgent(IsaacAgent):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.adaptive_task = self.env_cfg["task"]["adaptive_task"]

    def train_episode(self, phase=None, gui_app=None, gui_rew=None):
        episode_r, episode_steps, info = super().train_episode(gui_app, gui_rew)

        task_return = self.task.trainTaskR(episode_r)

        if self.adaptive_task:
            self.task.adapt_task()

        wandb.log(
            {
                f"reward/phase{phase}_train": self.game_rewards.get_mean(),
                f"reward/phase{phase}_episode_length": self.game_lengths.get_mean(),
            }
        )

        task_return = task_return.detach().tolist()
        for i in range(len(task_return)):
            wandb.log(
                {
                    f"reward/phase{phase}_task_return{i}": task_return[i],
                }
            )

        info["task_return"] = task_return
        return episode_r, episode_steps, info

    def evaluate(self, phase=None):
        returns, episode_r = super().evaluate()
        task_return = self.task.evalTaskR(episode_r)

        print("returns:", returns)
        print("task returns:", task_return)

        print(f"===== finish evaluate ====")

        wandb.log({f"reward/phase{phase}_eval": torch.mean(returns).item()})

        task_return = task_return.detach().tolist()
        for i in range(len(task_return)):
            wandb.log(
                {
                    f"reward/phase{phase}_task_return{i}": task_return[i],
                }
            )

        return returns, {"task_return": task_return}

    def _act(self, s, task, mode):
        with torch.no_grad():
            if (self.steps <= self.min_n_experience) and mode == "explore":
                a = (
                    2 * torch.rand((self.n_env, self.env.num_act), device=self.device)
                    - 1
                )

            w = copy.copy(np2ts(task.W))
            id = copy.copy(np2ts(task.id))

            if mode == "explore":
                a = self.explore(s, w, id)
            elif mode == "exploit":
                a = self.exploit(s, w, id)
        return a
