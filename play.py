import isaacgym

from omegaconf import DictConfig, OmegaConf
import argparse
import json
from common.util import (
    print_dict,
    AverageMeter,
)

from run import get_agent

import torch
import numpy as np

from tkinter import *

# Initialize parser
parser = argparse.ArgumentParser()

# # Adding optional argument
parser.add_argument("-p", "--path", help="model save path", type=str, required=True)
parser.add_argument(
    "-c",
    "--checkpoint",
    help="specific saved model e.g. model10",
    type=str,
    required=True,
)

# Read arguments from command line
args = parser.parse_args()


class PlayUI:
    def __init__(self, cfg_dict, model_path) -> None:
        self.root = Tk()
        self.root.title("test")
        self.root.geometry("1300x200")
        self.frame = Frame(self.root)
        self.frame.pack()

        # init and load agent
        self.agent = get_agent(cfg_dict)
        self.agent.load_torch_model(model_path)

        self.weights = self.agent.task.Eval.W.clone()

        self.weightLabels = cfg_dict["env"]["task"]["taskLabels"]

        self.rew = None
        self.rb_z = None
        self.rel_hov = None
        self.wp_idx = None

        self.generate_scales()
        self.print_rb_info()

    def weight_update_function(self, dimension):
        def update_val(val):
            self.weights[..., dimension] = float(val)
            self.agent.task.Eval.W[:] = self.weights[:]
            self.agent.task.Eval.W = (
                self.agent.task.Eval.W / self.agent.task.Eval.W.norm(1, 1, keepdim=True)
            )

        return update_val

    def target_update_function(self, dimension):
        def update_val(val):
            self.agent.env.wp.ang[..., 2] = float(val)

        return update_val

    def add_scale(self, dimension, gen_func, label, range=(-0.2, 1), type="weight"):
        scale = Scale(
            self.frame,
            from_=range[0],
            to=range[1],
            digits=3,
            resolution=0.01,
            label=label,
            orient=VERTICAL,
            command=gen_func(dimension),
        )
        if type == "weight":
            scale.set(self.agent.task.Eval.W[0, dimension].item())
        scale.pack(side=LEFT)

    def generate_scales(self):
        for i, label in enumerate(self.weightLabels):
            self.add_scale(
                dimension=i, gen_func=self.weight_update_function, label=label
            )

        self.add_scale(
            dimension=3,
            gen_func=self.target_update_function,
            label="target yaw",
            range=(-np.pi, np.pi),
            type="target",
        )

    def print_rb_info(self):
        self.rew = DoubleVar(name="reward")  # instantiate the IntVar variable class
        self.rew.set(0.0)  # set it to 0 as the initial value

        self.rb_z = DoubleVar(name="robot z")  # instantiate the IntVar variable class
        self.rb_z.set(0.0)  # set it to 0 as the initial value

        self.rel_hov = DoubleVar(
            name="hov dist"
        )  # instantiate the IntVar variable class
        self.rel_hov.set(0.0)  # set it to 0 as the initial value

        self.wp_idx = DoubleVar(name="wp idx")  # instantiate the IntVar variable class
        self.wp_idx.set(0)  # set it to 0 as the initial value

        # the label's textvariable is set to the variable class instance
        Label(self.root, text="step reward: ").pack(side=LEFT)
        Label(self.root, textvariable=self.rew).pack(side=LEFT)

        Label(self.root, text="robot z: ").pack(side=LEFT)
        Label(self.root, textvariable=self.rb_z).pack(side=LEFT)

        Label(self.root, text="hov dist: ").pack(side=LEFT)
        Label(self.root, textvariable=self.rel_hov).pack(side=LEFT)

        Label(self.root, text="wp idx: ").pack(side=LEFT)
        Label(self.root, textvariable=self.wp_idx).pack(side=LEFT)

    def _debug_ui(self):
        # only runs UI loop without inference
        while True:
            self.root.update_idletasks()
            self.root.update()

    def play(self):
        print("self.agent.task.Eval:", self.agent.task.Eval.W)
        avgStepRew = AverageMeter(1, 20).to(self.agent.device)
        rec_pos_x = []
        rec_pos_y = []
        rec_pos_z = []
        rec_ang_z = []
        while True:
            s = self.agent.reset_env()
            for _ in range(2000):
                self.root.update_idletasks()
                self.root.update()

                a = self.agent.act(s, self.agent.task.Eval, "exploit")
                a[:, 2] = 1

                self.agent.env.step(a)
                s_next = self.agent.env.obs_buf.clone()
                self.agent.env.reset()

                r = self.agent.calc_reward(s_next, self.agent.task.Eval.W)
                s = s_next
                avgStepRew.update(r)

                if self.rew:
                    self.rew.set(avgStepRew.get_mean())

                ang_z = s[:, 2]
                pos_x = s[:, 11]
                pos_y = s[:, 12]
                pos_z = s[:, 13] + 20

                rec_ang_z.append(ang_z)
                rec_pos_x.append(pos_x)
                rec_pos_y.append(pos_y)
                rec_pos_z.append(pos_z)

            rec_ang_z = torch.stack(rec_ang_z).squeeze().cpu().numpy()
            rec_pos_x = torch.stack(rec_pos_x).squeeze().cpu().numpy()
            rec_pos_y = torch.stack(rec_pos_y).squeeze().cpu().numpy()
            rec_pos_z = torch.stack(rec_pos_z).squeeze().cpu().numpy()
            np.savetxt("text_angz.out", [rec_ang_z], delimiter=",", newline="")
            np.savetxt("text_posx.out", [rec_pos_x], delimiter=",", newline="")
            np.savetxt("text_posy.out", [rec_pos_y], delimiter=",", newline="")
            np.savetxt("text_posz.out", [rec_pos_z], delimiter=",", newline="")

            break


def modify_cfg(cfg_dict):
    # don't change these
    cfg_dict["agent"]["use_decoder"] = False
    cfg_dict["agent"]["load_model"] = False
    cfg_dict["env"]["save_model"] = False
    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    cfg_dict["buffer"]["min_n_experience"] = 0
    cfg_dict["env"]["task"]["rand_task"] = False
    cfg_dict["env"]["task"]["rand_vel_targets"] = False
    cfg_dict["env"]["mode"] = "play"
    cfg_dict["env"]["sim"]["headless"] = False
    cfg_dict["env"]["goal"]["target_velnorm"] = 2.5
    cfg_dict["env"]["blimp"]["reset_dist"] = 40
    # cfg_dict["env"]["episode_max_step"] = int(50 * (512 / cfg_dict["env"]["num_envs"]))

    # change these
    cfg_dict["agent"]["phase"] = 1  # phase: [encoder, adaptor, fine-tune, deploy]
    # cfg_dict["agent"]["name"] = "PID"
    cfg_dict["env"]["num_envs"] = 1
    cfg_dict["env"]["goal"]["type"] = "fix"
    cfg_dict["env"]["goal"]["style"] = "square"  # square, hourglass, circle
    cfg_dict["env"]["goal"]["trigger_dist"] = 5.5
    cfg_dict["env"]["task"]["task_wEval"] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # ['planar','Z','trigger','heading',  'proximity','yaw','vnorm',   'vxy','vz',  'bndcost','regRP','regT', 'regS']
    if "aero" in cfg_dict["env"]:
        cfg_dict["env"]["aero"]["wind_mag"] = 0.1
    if "domain_rand" in cfg_dict["env"]["task"]:
        cfg_dict["env"]["task"]["domain_rand"] = True
    cfg_dict["agent"]["exploit_method"] = "sfgpi"  # sfgpi, dacgpi

    print_dict(cfg_dict)

    return cfg_dict


# @hydra.main(config_name="config", config_path="./cfg")
# def launch_rlg_hydra(cfg: DictConfig):
#     cfg_dict = omegaconf_to_dict(cfg)

#     wandb.init(mode="disabled")
#     wandb_dict = fix_wandb(wandb.config)

#     model_folder = ""
#     model_checkpoint = "model980"

#     cfg_path = model_folder + "/cfg"
#     model_path = model_folder + "/" + model_checkpoint

#     cfg_dict = None
#     with open(cfg_path) as f:
#         cfg_dict = json.load(f)

#     # print_dict(wandb_dict)
#     update_dict(cfg_dict, wandb_dict)

#     cfg_dict = modify_cfg(cfg_dict)

#     playob = PlayUI(cfg_dict, model_path)
#     playob.play()

#     wandb.finish()


def launch_play():
    model_folder = args.path
    model_checkpoint = args.checkpoint

    cfg_path = model_folder + "/cfg"
    model_path = model_folder + "/" + model_checkpoint + "/"

    cfg_dict = None
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    cfg_dict = modify_cfg(cfg_dict)
    print(cfg_dict)

    playob = PlayUI(cfg_dict, model_path)
    playob.play()


if __name__ == "__main__":
    torch.manual_seed(456)
    np.random.seed(456)

    launch_rlg_hydra()
    # launch_play()
