import torch
from torch.optim import Adam

import wandb

from common.agent import IsaacAgent
from common.pid import (
    PIDController,
    BlimpPositionControl,
    BlimpHoverControl,
    BlimpVelocityControl,
)
from common.torch_jit_utils import *


class BlimpPositionController(IsaacAgent):
    ctrl_cfg = {
        "yaw": {
            "pid_param": torch.tensor([1.0, 0.01, 0.025]),
            "gain": 1,
        },
        "alt": {
            "pid_param": torch.tensor([1.0, 0.01, 0.5]),
            "gain": 0.2,
        },
        "vel": {
            "pid_param": torch.tensor([0.7, 0.01, 0.5]),
            "gain": 0.01,
        },
    }

    def __init__(self, cfg):
        super().__init__(cfg)

        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=self.device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=self.device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=self.device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_rb_angle = slice(0, 0 + 3)
        self.slice_goal_angle = slice(3, 3 + 3)
        self.slice_err_posNav = slice(8, 8 + 3)
        self.slice_err_posHov = slice(11, 11 + 3)

        self.slice_rb_v = slice(14, 14 + 3)
        self.slice_goal_v = slice(17, 17 + 3)

    # def explore(self, s, w): # position control
    #     err_yaw, err_planar, err_z = self.parse_state(s)

    #     yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
    #     alt_ctrl = self.alt_ctrl.action(err_z)
    #     vel_ctrl = self.vel_ctrl.action(err_planar)
    #     thrust_vec = -1 * torch.ones_like(vel_ctrl)
    #     a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
    #     # a = torch.tensor([-1, 0, thrust_vec, 0])

    #     return a

    # def explore(self, s, w): # hover control
    #     err_heading, err_planar, err_z = self.parse_state(s)

    #     yaw_ctrl = -self.yaw_ctrl.action(err_heading)
    #     alt_ctrl = self.alt_ctrl.action(err_z)

    #     vel_ctrl = torch.where(
    #         err_z[:, None] <= -3,
    #         torch.ones_like(alt_ctrl),
    #         self.vel_ctrl.action(err_planar),
    #     )
    #     thrust_vec = torch.where(
    #         err_z[:, None] <= -3,
    #         torch.zeros_like(vel_ctrl),
    #         -1 * torch.ones_like(vel_ctrl),
    #     )

    #     a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
    #     return a

    def explore(self, s, w): # velocity control
        err_vx, err_vy, err_vz, err_z = self.parse_state(s)

        yaw_ctrl = self.yaw_ctrl.action(err_vy)
        alt_ctrl = 5 * self.alt_ctrl.action(err_vz)

        vel_ctrl = torch.where(
            err_z[:, None] <= -3,
            torch.ones_like(alt_ctrl),
            -self.vel_ctrl.action(err_vx),
        )
        thrust_vec = torch.where(
            err_z[:, None] <= -3,
            torch.zeros_like(vel_ctrl),
            -1 * torch.ones_like(vel_ctrl),
        )
        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
        return a

    # def explore(self, s, w): # backward flight
    #     err_heading, err_planar, err_z = self.parse_state(s)

    #     yaw_ctrl = -self.yaw_ctrl.action(err_heading)
    #     alt_ctrl = -self.alt_ctrl.action(err_z)

    #     vel_ctrl = torch.where(
    #         err_z[:, None] <= -3,
    #         torch.ones_like(alt_ctrl),
    #         torch.clip(5*self.vel_ctrl.action(err_planar)-1, -1, -.2),
    #     )
    #     thrust_vec = torch.where(
    #         err_z[:, None] <= -3,
    #         torch.zeros_like(vel_ctrl),
    #         torch.ones_like(vel_ctrl),
    #     )
    #     a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
    #     return a


    def exploit(self, s, w):
        return self.explore(s, w)

    # def parse_state(self, s): # backward flight
    #     error_posNav = s[:, self.slice_err_posNav]
    #     robot_angle = s[:, self.slice_rb_angle]

    #     error_navHeading = check_angle(
    #         compute_heading(yaw=robot_angle[:, 2], rel_pos=error_posNav) +torch.pi
    #     )
    #     err_planar = error_posNav[:, 0:2]
    #     err_planar = torch.norm(err_planar, dim=1, keepdim=True)
    #     err_z = error_posNav[:, 2]
    #     return error_navHeading, err_planar, err_z

    # def parse_state(self, s): # position control
    #     error_posNav = s[:, self.slice_err_posNav]
    #     robot_angle = s[:, self.slice_rb_angle]

    #     error_navHeading = check_angle(
    #         compute_heading(yaw=robot_angle[:, 2], rel_pos=error_posNav)
    #     )
    #     err_planar = error_posNav[:, 0:2]
    #     err_planar = torch.norm(err_planar, dim=1, keepdim=True)
    #     err_z = error_posNav[:, 2]
    #     return error_navHeading, err_planar, err_z

    # def parse_state(self, s): # hover control
    #     error_posHov = s[:, self.slice_err_posHov]
    #     robot_angle = s[:, self.slice_rb_angle]

    #     error_hovHeading = check_angle(
    #         compute_heading(yaw=robot_angle[:, 2], rel_pos=error_posHov)
    #     )
    #     err_planar = error_posHov[:, 0:2]
    #     err_planar = torch.norm(err_planar, dim=1, keepdim=True)
    #     err_z = error_posHov[:, 2]
    #     return error_hovHeading, err_planar, err_z

    def parse_state(self, s): # velocity control
        rb_v = s[:, self.slice_rb_v]
        goal_v = s[:, self.slice_goal_v]
        robot_angle = s[:, self.slice_rb_angle]
        robot_angvel = s[:, 21:21+3]

        error_posNav = s[:, self.slice_err_posNav]
        err_z = error_posNav[:, 2]

        error_v = rb_v - goal_v
        robot_angle = s[:, self.slice_rb_angle]

        err_vx, error_vy, error_vz = globalToLocalRot(
            robot_angle[:, 0],
            robot_angle[:, 1],
            robot_angle[:, 2],
            error_v[:, 0],
            error_v[:, 1],
            error_v[:, 2],
        )

        return err_vx, error_vy, error_vz, err_z

    def learn(self):
        pass

    def clear(self):
        self.yaw_ctrl.clear()
        self.alt_ctrl.clear()
        self.vel_ctrl.clear()
