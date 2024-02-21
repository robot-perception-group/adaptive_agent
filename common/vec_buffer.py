import torch

try:
    from tensordict import TensorDict
except:
    print("no tensordict")

try:
    from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
except:
    print("no torchrl")


class VectorizedReplayBuffer:
    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        device,
        mini_batch_size=64,
        add_task_ids=False,
        add_task_weights=False,
        task_id_dim=1,
        task_weight_dim=13,
        *args,
        **kwargs,
    ):
        """Create Vectorized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """

        self.device = device
        self.batch_size = mini_batch_size
        self.add_task_ids = add_task_ids
        self.add_task_weights = add_task_weights

        self.obses = torch.empty(
            (capacity, *obs_shape), dtype=torch.float32, device=self.device
        )
        self.next_obses = torch.empty(
            (capacity, *obs_shape), dtype=torch.float32, device=self.device
        )
        self.actions = torch.empty(
            (capacity, *action_shape), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.empty(
            (capacity, 1), dtype=torch.float32, device=self.device
        )
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device=self.device)

        if self.add_task_ids:
            self.task_ids = torch.empty(
                (capacity, task_id_dim), dtype=torch.float32, device=self.device
            )
        if self.add_task_weights:
            self.task_weights = torch.empty(
                (capacity, task_weight_dim), dtype=torch.float32, device=self.device
            )

        self.capacity = capacity
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.idx

    def add(self, obs, action, reward, next_obs, done, id=None, weight=None):
        num_observations = obs.shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            self.obses[0:overflow] = obs[-overflow:]
            self.actions[0:overflow] = action[-overflow:]
            self.rewards[0:overflow] = reward[-overflow:]
            self.next_obses[0:overflow] = next_obs[-overflow:]
            self.dones[0:overflow] = done[-overflow:]
            if self.add_task_ids:
                self.task_ids[0:overflow] = id[-overflow:]
            if self.add_task_weights:
                self.task_weights[0:overflow] = weight[-overflow:]
            self.full = True
        self.obses[self.idx : self.idx + remaining_capacity] = obs[:remaining_capacity]
        self.actions[self.idx : self.idx + remaining_capacity] = action[
            :remaining_capacity
        ]
        self.rewards[self.idx : self.idx + remaining_capacity] = reward[
            :remaining_capacity
        ]
        self.next_obses[self.idx : self.idx + remaining_capacity] = next_obs[
            :remaining_capacity
        ]
        self.dones[self.idx : self.idx + remaining_capacity] = done[:remaining_capacity]

        if self.add_task_ids:
            self.task_ids[self.idx : self.idx + remaining_capacity] = id[
                :remaining_capacity
            ]
        if self.add_task_weights:
            self.task_weights[self.idx : self.idx + remaining_capacity] = weight[
                :remaining_capacity
            ]

        self.idx = (self.idx + num_observations) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size=None, id=None):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obses: torch tensor
            batch of observations
        actions: torch tensor
            batch of actions executed given obs
        rewards: torch tensor
            rewards received as results of executing act_batch
        next_obses: torch tensor
            next set of observations seen after executing act_batch
        not_dones: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not
        not_dones_no_max: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not, specifically exlcuding maximum episode steps
        """
        if batch_size is None:
            batch_size = self.batch_size

        idxs = torch.randint(
            0,
            self.capacity if self.full else self.idx,
            (batch_size,),
            device=self.device,
        )
        ids = None
        if self.add_task_ids:
            ids = self.task_ids[idxs]

        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        dones = self.dones[idxs]

        weights = None
        if self.add_task_weights:
            weights = self.task_weights[idxs]

        if self.add_task_ids and id!=None:
            msk = (ids == id)

            obses = obses[msk.squeeze(), :]
            actions = actions[msk.squeeze(), :]
            rewards = rewards[msk.squeeze(), :]
            next_obses = next_obses[msk.squeeze(), :]
            dones = dones[msk.squeeze(), :]
            ids = ids[msk.squeeze(), :]

            if self.add_task_weights:
                weights = weights[msk.squeeze(), :]

        return {
            "obs": obses,
            "action": actions,
            "reward": rewards,
            "next_obs": next_obses,
            "done": dones,
            "ids": ids,
            "weights": weights,
        }


class FrameStackedReplayBuffer:
    def __init__(
        self,
        obs_shape,
        action_shape,
        capacity,
        n_env,
        stack_size,
        device,
        add_task_weights=False,
        task_weight_dim=None,
        mini_batch_size=64,
        *args,
        **kwargs,
    ):
        self.device = device
        self.batch_size = mini_batch_size
        self.capacity = int(capacity // n_env)

        self.add_task_weights = add_task_weights

        self.n_env = int(n_env)
        self.stack_size = stack_size

        self.obses = torch.empty(
            (self.capacity, n_env, *obs_shape), dtype=torch.float32, device=self.device
        )
        self.next_obses = torch.empty(
            (self.capacity, n_env, *obs_shape), dtype=torch.float32, device=self.device
        )
        self.actions = torch.empty(
            (self.capacity, n_env, *action_shape),
            dtype=torch.float32,
            device=self.device,
        )
        self.rewards = torch.empty(
            (self.capacity, n_env, 1), dtype=torch.float32, device=self.device
        )
        self.dones = torch.empty(
            (self.capacity, n_env, 1), dtype=torch.bool, device=self.device
        )

        if self.add_task_weights:
            self.task_weights = torch.empty(
                (self.capacity, n_env, task_weight_dim), dtype=torch.float32, device=self.device
            )


        self.idx = 0
        self.full = False

        self.stack_range = torch.arange(0, self.stack_size, device=self.device)

    def __len__(self):
        return self.idx * self.n_env

    def add(self, obs, action, reward, next_obs, done, weight=None):
        if self.idx == self.capacity:
            self.full = True
            self.idx = 0

        self.obses[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obses[self.idx] = next_obs
        self.dones[self.idx] = done

        if self.add_task_weights:
            self.task_weights[self.idx] = weight

        self.idx += 1

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        # select sample
        if self.full:
            sra = [0, self.capacity]
        else:
            sra = [0, self.idx]

        idx1 = torch.randint(
            sra[0],
            sra[1],
            (batch_size,),
            device=self.device,
        )

        # select env
        idx2 = torch.randint(
            0,
            self.n_env,
            (batch_size,),
            device=self.device,
        )

        obses = self.obses[idx1, idx2]  # [B, F] <-- [N, NE, F]
        stacked_obs = self.stack_data(self.obses, idx1, idx2)  # [B, F, S]
        actions = self.actions[idx1, idx2]  # [B,A]
        stacked_act = self.stack_data(self.actions, idx1, idx2)  # [B, F, S]
        rewards = self.rewards[idx1, idx2]
        next_obses = self.next_obses[idx1, idx2]
        dones = self.dones[idx1, idx2]

        task_weights = None
        if self.add_task_weights:
            task_weights = self.task_weights[idx1, idx2]

        return {
            "obs": obses,
            "stacked_obs": stacked_obs,
            "action": actions,
            "stacked_act": stacked_act,
            "reward": rewards,
            "next_obs": next_obses,
            "done": dones,
            "weights" : task_weights,
        }

    def stack_data(self, x, idx1, idx2):
        ra = self.stack_range.repeat((idx1.shape[0], 1))  # [B, S]
        ids1 = idx1[:, None] - ra  # [B, S] <-- [B]
        ids2 = idx2[:, None].repeat_interleave(self.stack_size, 1)  # [B, S] <-- [B]

        stacked_obj = x[ids1, ids2]  # [B, S, F] <-- [N, NE, F]

        if not self.full:
            stacked_obj[ids1 < 0] = 0

        # correct by dones
        dones = self.dones[ids1, ids2]  # [B, S, 1] <-- [N, NE, 1]
        dones[:, 0] = False
        mask = torch.where(torch.cumsum(dones, 1) > 0, 0, 1)  # [B, S, 1]

        stacked_obj = mask * stacked_obj  # [B, S, F]=[B, S, 1]*[B, S, F]

        return stacked_obj.permute(0, 2, 1)  # [B, F, S]


class VecPrioritizedReplayBuffer:
    def __init__(
        self,
        capacity,
        device,
        alpha=0.6,
        beta=0.4,
        mini_batch_size=64,
        *args,
        **kwargs,
    ):
        self.device = device
        self.rb = TensorDictPrioritizedReplayBuffer(
            alpha=alpha,
            beta=beta,
            storage=LazyTensorStorage(capacity, device=self.device),
            batch_size=mini_batch_size,
        )

    def add(self, obs, action, reward, next_obs, done):
        data = TensorDict(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
            },
            obs.shape[0],
        )
        self.rb.extend(data)

    def sample(self, ndata=None):
        if ndata is not None:
            d = self.rb.sample(ndata)
        else:
            d = self.rb.sample()
        return d

    def update_tensordict_priority(self, sample):
        self.rb.update_tensordict_priority(sample)

    def __len__(self):
        return len(self.rb)


if __name__ == "__main__":
    capacity = 20
    device = "cuda"
    data_size = 5
    stack_size = 3

    n_env = 2
    obs_dim = 4
    feat_dim = 4
    act_dim = 2
    rew_dim = 1
    done_dim = 1

    buf = FrameStackedReplayBuffer(
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
        capacity=capacity,
        n_env=n_env,
        stack_size=stack_size,
        device=device,
    )

    for _ in range(data_size):
        obs = torch.rand(n_env, obs_dim)
        action = torch.rand(n_env, act_dim)
        reward = torch.rand(n_env, rew_dim)
        next_obs = torch.rand(n_env, obs_dim)
        done = torch.randint(0, 2, (n_env, done_dim))

        buf.add(obs, action, reward, next_obs, done)

    print("len buf", len(buf))

    sample = buf.sample(1)
    print("buf.obs", buf.obses)
    # print("buf.dones", buf.dones)
    # print("obs", sample["obs"])
    print("stacked_obs", sample["stacked_obs"])
    # print("obs shape", sample["obs"].shape)
    # print("stacked_obs shape", sample["stacked_obs"].shape)
