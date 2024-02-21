import torch
from common.torch_jit_utils import *


class PIDController:
    def __init__(
        self,
        pid_param,
        gain,
        device,
        offset=0.0,
        delta_t=0.01,
        i_from_sensor=False,
        d_from_sensor=False,
    ):
        self.pid_param = torch.tensor(pid_param, device=device)[:, None]
        self.gain = gain
        self.offset = offset
        self.delta_t = delta_t
        self.i_from_sensor = i_from_sensor
        self.d_from_sensor = d_from_sensor

        self.err_sum, self.prev_err = 0, 0

    def action(self, err, err_i=0, err_d=0):
        if err.dim() == 1:
            err = err[:, None]

        if not self.i_from_sensor:
            self.err_sum += err * self.delta_t
            self.err_sum = torch.clip(self.err_sum, -1, 1)
            err_i = self.err_sum

        if not self.d_from_sensor:
            err_d = (err - self.prev_err) / (self.delta_t)
            self.prev_err = err

        pid = torch.concat([err, err_i, err_d], dim=1)
        ctrl = self.gain * (pid @ self.pid_param)
        return ctrl + self.offset

    def clear(self):
        self.err_sum, self.prev_err = 0, 0


class BlimpPositionControl:
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
            "gain": 0.001,
        },
    }

    def __init__(self, device):
        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_rb_angle = slice(0, 0 + 3)
        self.slice_goal_angle = slice(3, 3 + 3)
        self.slice_err_posNav = slice(8, 8 + 3)

    def act_on_stack(self, s_stack, k=2):  # [N, S, K]
        self.clear()
        for i in reversed(range(k)):
            a = self.act(s_stack[:, :, -i - 1])
        return a

    # def act(self, s):
    #     err_yaw, err_planar, err_z = self.parse_state(s)

    #     yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
    #     alt_ctrl = self.alt_ctrl.action(err_z)
    #     vel_ctrl = self.vel_ctrl.action(err_planar)
    #     thrust_vec = -1 * torch.ones_like(vel_ctrl)
    #     a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
    #     return a

    def act(self, s):
        err_yaw, err_planar, err_z = self.parse_state(s)

        yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
        alt_ctrl = self.alt_ctrl.action(err_z)

        vel_ctrl = torch.where(
            err_z[:, None] <= -3,
            torch.ones_like(alt_ctrl),
            self.vel_ctrl.action(err_planar),
        )
        thrust_vec = torch.where(
            err_z[:, None] <= -3,
            torch.zeros_like(vel_ctrl),
            -1 * torch.ones_like(vel_ctrl),
        )

        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
        return a

    def parse_state(self, s):
        error_posNav = s[:, self.slice_err_posNav]
        robot_angle = s[:, self.slice_rb_angle]

        error_navHeading = check_angle(
            compute_heading(yaw=robot_angle[:, 2], rel_pos=error_posNav)
        )
        err_planar = error_posNav[:, 0:2]
        err_planar = torch.norm(err_planar, dim=1, keepdim=True)
        err_z = error_posNav[:, 2]
        return error_navHeading, err_planar, err_z

    def clear(self):
        self.yaw_ctrl.clear()
        self.alt_ctrl.clear()
        self.vel_ctrl.clear()


class BlimpHoverControl(BlimpPositionControl):
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
            "gain": 0.005,
        },
    }

    def __init__(self, device):
        super().__init__(device)

        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_rb_angle = slice(0, 0 + 3)
        self.slice_goal_angle = slice(3, 3 + 3)
        self.slice_err_posNav = slice(8, 8 + 3)
        self.slice_err_posHov = slice(11, 11 + 3)

    def act(self, s):
        err_yaw, err_planar, err_z = self.parse_state(s)

        yaw_ctrl = -self.yaw_ctrl.action(err_yaw)
        alt_ctrl = self.alt_ctrl.action(err_z)

        vel_ctrl = torch.where(
            err_z[:, None] <= -3,
            torch.ones_like(alt_ctrl),
            self.vel_ctrl.action(err_planar),
        )
        thrust_vec = torch.where(
            err_z[:, None] <= -3,
            torch.zeros_like(vel_ctrl),
            -1 * torch.ones_like(vel_ctrl),
        )

        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
        return a

    def parse_state(self, s):
        error_posHov = s[:, self.slice_err_posHov]
        robot_angle = s[:, self.slice_rb_angle]

        error_navHeading = check_angle(
            compute_heading(yaw=robot_angle[:, 2], rel_pos=error_posHov)
        )
        err_planar = error_posHov[:, 0:2]
        err_planar = torch.norm(err_planar, dim=1, keepdim=True)
        err_z = error_posHov[:, 2]
        return error_navHeading, err_planar, err_z


class BlimpVelocityControl(BlimpPositionControl):
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
            "gain": 1.0,
        },
    }

    def __init__(self, device):
        super().__init__(device)

        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_rb_angle = slice(0, 0 + 3)
        self.slice_goal_angle = slice(3, 3 + 3)
        self.slice_err_posNav = slice(8, 8 + 3)
        self.slice_err_posHov = slice(11, 11 + 3)
        self.slice_rb_v = slice(14, 14 + 3)
        self.slice_goal_v = slice(17, 17 + 3)

    def act(self, s):
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

    def parse_state(self, s):
        rb_v = s[:, self.slice_rb_v]
        goal_v = s[:, self.slice_goal_v]
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

class BlimpBackwardControl(BlimpPositionControl):
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
            "gain": 0.005,
        },
    }

    def __init__(self, device):
        super().__init__(device)

        delta_t = 0.1

        self.yaw_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["yaw"],
        )
        self.alt_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["alt"],
        )
        self.vel_ctrl = PIDController(
            device=device,
            delta_t=delta_t,
            **self.ctrl_cfg["vel"],
        )

        self.slice_rb_angle = slice(0, 0 + 3)
        self.slice_goal_angle = slice(3, 3 + 3)
        self.slice_err_posNav = slice(8, 8 + 3)
        self.slice_err_posHov = slice(11, 11 + 3)

    def act(self, s):
        err_heading, err_planar, err_z = self.parse_state(s)

        yaw_ctrl = -self.yaw_ctrl.action(err_heading)
        alt_ctrl = -self.alt_ctrl.action(err_z)

        vel_ctrl = torch.where(
            err_z[:, None] <= -3,
            torch.ones_like(alt_ctrl),
            torch.clip(5*self.vel_ctrl.action(err_planar)-1, -1, -.3),
        )
        thrust_vec = torch.where(
            err_z[:, None] <= -3,
            torch.zeros_like(vel_ctrl),
            torch.ones_like(vel_ctrl),
        )
        a = torch.concat([vel_ctrl, yaw_ctrl, thrust_vec, alt_ctrl], dim=1)
        return a

    def parse_state(self, s):
        error_posNav = s[:, self.slice_err_posNav]
        robot_angle = s[:, self.slice_rb_angle]

        error_navHeading = check_angle(
            compute_heading(yaw=robot_angle[:, 2], rel_pos=error_posNav) + torch.pi
        )
        err_planar = error_posNav[:, 0:2]
        err_planar = torch.norm(err_planar, dim=1, keepdim=True)
        err_z = error_posNav[:, 2]
        return error_navHeading, err_planar, err_z
