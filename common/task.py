import torch
import itertools


class TaskObject:
    def __init__(self, initW, n_env, randTasks, taskSetType, task_cfg, device) -> None:
        self.n_env = n_env
        self.randTasks = randTasks
        self.taskSetType = taskSetType
        self.task_cfg = task_cfg
        self.device = device

        self.initW = torch.tensor(initW, device=self.device, dtype=torch.float32)
        self.dim = int(self.initW.shape[0])

        self.W = self.normalize_task(torch.tile(self.initW, (self.n_env, 1)))  # [N, F]
        self.id = None

        if self.taskSetType != "uniform":
            self.taskSet = self.define_taskSet(self.taskSetType)
            assert self.taskSet.shape[1] == self.dim
            self.reset_taskRatio()

        if self.randTasks:
            self.W = self.sample_tasks()

    def define_taskSet(self, taskSetType):
        tasksets = self.task_cfg["taskSet"]

        if taskSetType == "permute":
            taskSet = list(itertools.product([0, 1], repeat=self.dim))
            taskSet.pop(0)  # remove all zero vector
        elif taskSetType == "identity":
            taskSet = [
                [1 if i == j else 0 for j in range(self.dim)] for i in range(self.dim)
            ]
        else:
            taskSet = tasksets[taskSetType]

        return torch.tensor(taskSet, dtype=torch.float32, device=self.device)

    def sample_tasks(self):
        if self.taskSetType == "uniform":
            tasks = torch.rand((self.n_env, self.dim), device=self.device)
        else:
            id = self.sample_taskID(self.taskRatio)

            assert len(self.taskSet) <= len(
                id
            ), f"num envs {len(id)} is less than num taskSet {len(self.taskSet)}"

            # ensure that all task are in id
            for i in range(len(self.taskSet)):
                if i not in id:
                    id[i] = i

            self.update_id(id)
            tasks = self.taskSet[id]

        return self.normalize_task(tasks)

    def update_id(self, id):
        self.id = id

    def normalize_task(self, w):
        w /= (w.norm(1, 1, keepdim=True)+1e-7)
        return w

    def sample_taskID(self, ratio):
        """sample tasks based on their ratio"""
        return torch.multinomial(ratio, self.n_env, replacement=True)

    def reset_taskRatio(self):
        self.taskRatio = torch.ones(len(self.taskSet), device=self.device) / len(
            self.taskSet
        )

    def add_task(self, w: torch.tensor):
        w = w.view(-1, self.dim)
        self.taskSet = torch.cat([self.taskSet, w], 0)
        self.reset_taskRatio()
        self.W = self.sample_tasks()


class SmartTask:
    def __init__(self, env_cfg, device) -> None:
        self.env_cfg = env_cfg
        self.task_cfg = self.env_cfg["task"]
        self.feature_cfg = self.env_cfg["feature"]
        self.device = device
        self.verbose = self.task_cfg.get("verbose", False)

        self.n_env = self.env_cfg["num_envs"]
        self.use_feature = self.feature_cfg.get("use_feature", None)
        self.randTasks = self.task_cfg.get("rand_task", False)
        self.taskSet_train = self.task_cfg.get("taskSet_train", None)
        self.taskSet_eval = self.task_cfg.get("taskSet_eval", None)
        self.intervalWeightRand = self.task_cfg.get("intervalWeightRand", 2)

        wTrain = self.define_task(self.use_feature, self.task_cfg["task_wTrain"])
        wEval = self.define_task(self.use_feature, self.task_cfg["task_wEval"])

        self.Train = TaskObject(
            wTrain,
            self.n_env,
            self.randTasks,
            self.taskSet_train,
            self.task_cfg,
            device,
        )
        self.Eval = TaskObject(
            wEval, self.n_env, self.randTasks, self.taskSet_eval, self.task_cfg, device
        )

        self.dim = int(self.Train.dim)

        if self.verbose:
            print("[Task] training task set: \n", self.Train.taskSetType)
            print("[Task] training tasks id: \n", self.Train.id)
            print("[Task] training tasks: \n", self.Train.W)
            print("[Task] evaluation task set: \n", self.Eval.taskSetType)
            print("[Task] evaluation tasks id: \n", self.Eval.id)
            print("[Task] evaluation tasks: \n", self.Eval.W)
            print("\n")

    def define_task(self, c, w):
        """define initial task weight as a vector"""
        l = []
        for i in range(len(w)):
            l += c[i] * [w[i]]
        return l

    def rand_task(self, episodes):
        if (
            ((episodes - 1) % self.intervalWeightRand == 0)
            and (self.env_cfg["mode"] == "train")
            and (self.randTasks)
        ):
            self.Train.sample_tasks()

            if self.verbose:
                print("[Task] sample new tasks:")
                print("[Task] Train.W[0]: ", self.Train.W)
                print("[Task] Train.taskRatio: ", self.Train.taskRatio)
                print("[Task] Train Task Counts: ", torch.bincount(self.Train.id))
                print("[Task] Eval.W[0]: ", self.Eval.W[0])
                print("[Task] Eval.taskRatio: ", self.Eval.taskRatio)
                print("[Task] Eval Task Counts: ", torch.bincount(self.Eval.id))
                print("\n")

    def trainTaskR(self, episode_r):
        self.trainTaskReturn = self.get_taskR(self.Train, episode_r)
        return self.trainTaskReturn

    def evalTaskR(self, episode_r):
        self.EvalTaskReturn = self.get_taskR(self.Eval, episode_r)
        return self.EvalTaskReturn

    def get_taskR(self, taskObj, episode_r):
        taskRatio = taskObj.taskRatio
        id = taskObj.id
        return taskRatio.index_add(
            dim=0, index=id, source=episode_r.float()
        ) / torch.bincount(id)

    def adapt_task(self):
        """
        Update task ratio based on reward.
        The more reward the less likely for a task to be sampled.
        """
        new_ratio = self.trainTaskReturn**-1
        new_ratio /= new_ratio.norm(1, keepdim=True)
        self.Train.taskRatio = new_ratio

        if self.verbose:
            print(
                f"[Task] updated task ratio: {new_ratio} \n as inverse of return {self.trainTaskReturn} \n"
            )

    def add_task(self, w: torch.tensor):
        self.Train.add_task(w)
        if self.verbose:
            print(f"[Task] new task {w} added to train task set \n")


class PointMassTask(SmartTask):
    def __init__(self, env_cfg, device) -> None:
        self.env_dim = env_cfg["feature"]["dim"]

        super().__init__(env_cfg, device)

    def define_task(self, c, w):
        w_pos_norm = c[0] * [w[0]]
        w_vel = c[1] * self.env_dim * [w[1]]
        w_vel_norm = c[2] * [w[2]]
        w_prox = c[3] * [w[3]]
        return w_pos_norm + w_vel + w_vel_norm + w_prox


class PointerTask(SmartTask):
    def __init__(self, env_cfg, device) -> None:
        super().__init__(env_cfg, device)


class BlimpTask(SmartTask):
    def __init__(self, env_cfg, device) -> None:
        super().__init__(env_cfg, device)

    def define_task(self, c, w):
        return w


class AntTask(SmartTask):
    def __init__(self, env_cfg, device) -> None:
        super().__init__(env_cfg, device)

    def define_task(self, c, w):
        w_px = c[0] * [w[0]]
        w_py = c[1] * [w[1]]
        w_alive = c[2] * [w[2]]

        return w_px + w_py + w_alive


def task_constructor(env_cfg, device):
    if "pointer" in env_cfg["env_name"].lower():
        return PointerTask(env_cfg, device)
    elif "pointmass" in env_cfg["env_name"].lower():
        return PointMassTask(env_cfg, device)
    elif "blimp" in env_cfg["env_name"].lower():
        return BlimpTask(env_cfg, device)
    elif "ant" in env_cfg["env_name"].lower():
        return AntTask(env_cfg, device)
    else:
        print(f'task not implemented: {env_cfg["env_name"]}')
        return None
