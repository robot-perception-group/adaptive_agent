import math
import sys

import torch
from common.torch_jit_utils import *
from env.base.vec_env import VecEnv
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from common.pid import BlimpPositionControl, BlimpHoverControl, BlimpVelocityControl, BlimpBackwardControl

from .base.vec_env import VecEnv
from .base.goal import RandomWayPoints, FixWayPoints


class BlimpRand(VecEnv):
    def __init__(self, cfg):
        # task-specific parameters
        self.num_obs = 37
        self.num_act = 4

        # domain randomization
        self.num_latent = 31 
        self.num_obs += self.num_latent

        # export actions
        self.num_expert = 16
        self.num_obs += self.num_expert

        super().__init__(cfg=cfg)

        self.pos_lim = cfg["goal"].get("pos_lim", 20)
        self.vel_lim = cfg["goal"].get("vel_lim", 5)
        self.avel_lim = cfg["goal"].get("avel_lim", 0.5)
        self.reset_dist = cfg["blimp"].get("reset_dist", 40)  # when to reset [m]
        self.spawn_height = cfg["blimp"].get("spawn_height", 15)

        self.id_absZ = 0
        self.id_relpos = [0, 0, 0]
        self.id_ang = [0, 0]
        self.id_angvel = [0, 0, 0]

        # blimp parameters
        # smoothning factor for fan thrusts
        self.ema_smooth = torch.tensor(
            cfg["blimp"]["ema_smooth"], device=self.sim_device
        )
        self.drag_bodies = torch.tensor(
            cfg["blimp"]["drag_body_idxs"], device=self.sim_device
        ).to(torch.long)
        self.body_areas = torch.tensor(cfg["blimp"]["areas"], device=self.sim_device)
        self.drag_coefs = torch.tensor(
            cfg["blimp"]["drag_coef"], device=self.sim_device
        )
        self.blimp_mass = cfg["blimp"]["mass"]
        self.body_torque_coeff = torch.tensor(
            [0.47, 1.29, 270.0, 10.0, 4], device=self.sim_device
        )  # [coef, p, BL4, balance torque, fin torque coef]

        self.effort_thrust = 10.0  # 5.0
        self.effort_botthrust = 3.0

        # wind
        self.wind_dirs = torch.tensor(cfg["aero"]["wind_dirs"], device=self.sim_device)
        self.wind_mag = cfg["aero"]["wind_mag"]
        self.wind_std = cfg["aero"]["wind_std"]

        # randomized env latents
        self.domain_rand = cfg["task"].get("domain_rand", True)

        if self.domain_rand:
            range_a = cfg["task"].get("range_a", [0.8, 1.25])
            range_b = cfg["task"].get("range_b", [0.8, 1.25])
        else:
            range_a = [1.0, 1.0]
            range_b = [1.0, 1.0]

        self.set_latent_range(range_a, range_b)

        self.k_effort_thrust = torch.zeros(self.num_envs, device=self.sim_device)
        self.k_effort_botthrust = torch.zeros(self.num_envs, device=self.sim_device)
        self.k_ema_smooth = torch.zeros((self.num_envs, 2), device=self.sim_device)
        self.k_body_areas = torch.zeros((self.num_envs, 9, 3), device=self.sim_device)
        self.k_drag_coefs = torch.zeros((self.num_envs, 9, 3), device=self.sim_device)
        self.k_wind_mean = torch.zeros((self.num_envs, 3), device=self.sim_device)
        self.k_wind_std = torch.zeros((self.num_envs, 3), device=self.sim_device)
        self.k_wind = torch.zeros((self.num_envs, 3), device=self.sim_device)
        self.k_blimp_mass = torch.zeros(self.num_envs, device=self.sim_device)
        self.k_bouyancy = torch.zeros(self.num_envs, device=self.sim_device)
        self.k_body_torque_coeff = torch.zeros(
            (self.num_envs, 5), device=self.sim_device
        )

        self.randomize_latent()

        if "fix" in cfg["goal"]["type"].lower():
            self.wp = FixWayPoints(
                device=self.device,
                num_envs=self.num_envs,
                trigger_dist=cfg["goal"].get("trigger_dist", 2),
                pos_lim=cfg["goal"].get("lim", 10),
                style=cfg["goal"]["style"],
            )
        else:
            self.wp = RandomWayPoints(
                device=self.sim_device,
                num_envs=self.num_envs,
                init_pos=cfg["goal"]["target_pos"],
                init_vel=cfg["goal"]["target_vel"],
                init_velnorm=cfg["goal"]["target_velnorm"],
                init_ang=cfg["goal"]["target_ang"],
                init_angvel=cfg["goal"]["target_angvel"],
                rand_pos=cfg["goal"].get("rand_pos_targets", True),
                rand_ang=cfg["goal"].get("rand_ang_targets", True),
                rand_vel=cfg["goal"].get("rand_vel_targets", True),
                rand_angvel=cfg["goal"].get("rand_avel_targets", True),
                pos_lim=cfg["goal"].get("lim", 20),
                vel_lim=cfg["goal"].get("vel_lim", 5),
                angvel_lim=cfg["goal"].get("avel_lim", 0.5),
                kWayPt=cfg["goal"].get("kWayPt", 2),
                wp_dist=cfg["goal"].get("wp_dist", 10),
                trigger_dist=cfg["goal"].get("trigger_dist", 2),
                min_z=cfg["goal"].get("min_z", 5),
                max_z=cfg["goal"].get("max_z", 40),
                reset_dist=cfg["blimp"].get("reset_dist", 30),
            )

        # initialise envs and state tensors
        self.envs = self.create_envs()

        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.prepare_sim(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.rb_states = gymtorch.wrap_tensor(rb_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.rb_pos = self.rb_states[:, 0:3].view(self.num_envs, self.num_bodies, 3)
        self.rb_rot = self.rb_states[:, 3:7].view(self.num_envs, self.num_bodies, 4)
        self.rb_lvels = self.rb_states[:, 7:10].view(self.num_envs, self.num_bodies, 3)
        self.rb_avels = self.rb_states[:, 10:13].view(self.num_envs, self.num_bodies, 3)

        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_act),
            device=self.sim_device,
            dtype=torch.float,
        )
        self.prev_actuator = torch.zeros(
            (self.num_envs, 3),
            device=self.sim_device,
            dtype=torch.float,
        )

        # storing tensors for visualisations
        self.actions_tensor = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.sim_device,
            dtype=torch.float,
        )
        # self.actions_tensor_prev = torch.zeros(
        #     (self.num_envs, self.num_bodies, 3),
        #     device=self.sim_device,
        #     dtype=torch.float,
        # )
        self.torques_tensor = torch.zeros(
            (self.num_envs, self.num_bodies, 3),
            device=self.sim_device,
            dtype=torch.float,
        )

        # hint from PID controllers
        self.controllers = []
        self.controllers.append(BlimpPositionControl(device=self.device))
        self.controllers.append(BlimpHoverControl(device=self.device))
        self.controllers.append(BlimpVelocityControl(device=self.device))
        self.controllers.append(BlimpBackwardControl(device=self.device))

        # step simulation to initialise tensor buffers
        self.reset()

    def set_latent_range(self, range_a, range_b):
        ra0, ra1 = range_a
        rb0, rb1 = range_b
        
        self.range_body_areas0 = [self.body_areas[0] * ra0, self.body_areas[0] * ra1]
        self.range_body_areas1 = [self.body_areas[1] * ra0, self.body_areas[1] * ra1]
        self.range_body_areas2 = [self.body_areas[2] * ra0, self.body_areas[2] * ra1]
        self.range_drag_coefs0 = [self.drag_coefs[0] * ra0, self.drag_coefs[0] * ra1]
        self.range_drag_coefs1 = [self.drag_coefs[1] * ra0, self.drag_coefs[1] * ra1]
        self.range_wind_mag = [self.wind_mag * ra0, self.wind_mag * ra1]
        self.range_wind_std = [self.wind_std * ra0, self.wind_std * ra1]
        self.range_blimp_mass = [self.blimp_mass * ra0, self.blimp_mass * ra1]
        self.range_body_torque_coeff = [
            self.body_torque_coeff * ra0,
            self.body_torque_coeff * ra1,
        ]

        self.range_effort_thrust = [self.effort_thrust * rb0, self.effort_thrust * rb1]
        self.range_effort_botthrust = [
            self.effort_botthrust * rb0,
            self.effort_botthrust * rb1,
        ]
        self.range_ema_smooth = [self.ema_smooth * rb0, self.ema_smooth * rb1]

    def create_envs(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.restitution = 1
        plane_params.static_friction = 0
        plane_params.dynamic_friction = 0
        self.gym.add_ground(self.sim, plane_params)

        # define environment space (for visualisation)
        spacing = self.pos_lim
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        # add blimp asset
        asset_root = "assets"
        asset_file = "blimp/urdf/blimp.urdf"
        asset_options = gymapi.AssetOptions()
        blimp_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

        self.num_dof = self.gym.get_asset_dof_count(blimp_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(blimp_asset)

        # define blimp pose
        pose = gymapi.Transform()
        pose.p.z = self.spawn_height  # generate the blimp h m from the ground
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # generate environments
        self.blimp_handles = []
        envs = []
        print(f"Creating {self.num_envs} environments.")
        for i in range(self.num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add blimp here in each environment
            blimp_handle = self.gym.create_actor(
                env_ptr, blimp_asset, pose, "pointmass", i, 1, 0
            )

            dof_props = self.gym.get_actor_dof_properties(env_ptr, blimp_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"].fill(1000.0)
            dof_props["damping"].fill(500.0)

            self.gym.set_actor_dof_properties(env_ptr, blimp_handle, dof_props)

            envs.append(env_ptr)
            self.blimp_handles.append(blimp_handle)

        return envs

    def get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        # refreshes the rb state tensor with new values
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        body_id = 0 # 0: base, 13: top fin
        d = 0

        # robot angle
        roll, pitch, yaw = get_euler_xyz(self.rb_rot[env_ids, body_id, :])
        self.id_ang[0]=d
        self.obs_buf[env_ids, d] = check_angle(roll)
        d += 1
        self.id_ang[1]=d
        self.obs_buf[env_ids, d] = check_angle(pitch)
        d += 1
        self.obs_buf[env_ids, d] = check_angle(yaw)

        # goal angles
        d += 1  # 3
        self.obs_buf[env_ids, d] = check_angle(self.wp.ang[env_ids, 0])
        d += 1
        self.obs_buf[env_ids, d] = check_angle(self.wp.ang[env_ids, 1])
        d += 1
        self.obs_buf[env_ids, d] = check_angle(self.wp.ang[env_ids, 2])

        # robot z
        d += 1  # 6
        self.id_absZ = d
        self.obs_buf[env_ids, d] = self.rb_pos[env_ids, body_id, 2]

        # trigger navigation goal
        trigger = self.wp.update_state(self.rb_pos[:, 0])
        d += 1  # 7
        self.obs_buf[env_ids, d] = trigger[env_ids, 0]

        # relative pos to navigation goal
        rel_pos = self.rb_pos[env_ids, 0] - self.wp.get_pos_nav()[env_ids]
        d += 1  # 8
        self.obs_buf[env_ids, d] = rel_pos[:, 0]
        d += 1
        self.obs_buf[env_ids, d] = rel_pos[:, 1]
        d += 1
        self.obs_buf[env_ids, d] = rel_pos[:, 2]

        # relative pos to hover goal
        rel_pos = self.rb_pos[env_ids, 0] - self.wp.pos_hov[env_ids]
        d += 1  # 11
        self.id_relpos[0] = d
        self.obs_buf[env_ids, d] = rel_pos[:, 0]
        d += 1
        self.id_relpos[1] = d
        self.obs_buf[env_ids, d] = rel_pos[:, 1]
        d += 1
        self.id_relpos[2] = d
        self.obs_buf[env_ids, d] = rel_pos[:, 2]

        # robot vel
        d += 1  # 14
        self.obs_buf[env_ids, d] = self.rb_lvels[env_ids, body_id, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.rb_lvels[env_ids, body_id, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.rb_lvels[env_ids, body_id, 2]

        # goal vel
        d += 1  # 17
        self.obs_buf[env_ids, d] = self.wp.vel[env_ids, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.wp.vel[env_ids, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.wp.vel[env_ids, 2]

        d += 1 # 20
        self.obs_buf[env_ids, d] = self.wp.velnorm[env_ids, 0]

        # robot angular velocities
        d += 1  # 21
        self.id_angvel[0] = d
        self.obs_buf[env_ids, d] = self.rb_avels[env_ids, body_id, 0]
        d += 1
        self.id_angvel[1] = d
        self.obs_buf[env_ids, d] = self.rb_avels[env_ids, body_id, 1]
        d += 1
        self.id_angvel[2] = d
        self.obs_buf[env_ids, d] = self.rb_avels[env_ids, body_id, 2]

        # goal ang vel
        d += 1  # 24
        self.obs_buf[env_ids, d] = self.wp.angvel[env_ids, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.wp.angvel[env_ids, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.wp.angvel[env_ids, 2]

        # relative pos to the next navigation goal
        rel_pos = self.rb_pos[env_ids, 0] - self.wp.get_pos_nav(self.wp.idx+1)[env_ids]
        d += 1  # 27
        self.obs_buf[env_ids, d] = rel_pos[:, 0]
        d += 1
        self.obs_buf[env_ids, d] = rel_pos[:, 1]
        d += 1
        self.obs_buf[env_ids, d] = rel_pos[:, 2]

        # prev actuators
        d += 1  # 30
        self.obs_buf[env_ids, d] = self.prev_actuator[env_ids, 0]  # thrust
        d += 1
        self.obs_buf[env_ids, d] = self.prev_actuator[env_ids, 1]  # stick
        d += 1
        self.obs_buf[env_ids, d] = self.prev_actuator[env_ids, 2]  # bot thrust

        # previous actions
        d += 1  # 33
        self.obs_buf[env_ids, d] = self.prev_actions[env_ids, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.prev_actions[env_ids, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.prev_actions[env_ids, 2]
        d += 1
        self.obs_buf[env_ids, d] = self.prev_actions[env_ids, 3]

        # ==== include env_latent to the observation ====#

        # robot actuator states
        # thrust vectoring angle
        ang = self.dof_pos[env_ids, 0]

        d += 1  
        self.obs_buf[env_ids, d] = ang

        d += 1
        self.obs_buf[env_ids, d] = torch.sin(ang)

        d += 1
        self.obs_buf[env_ids, d] = torch.cos(ang)

        # rudder
        d += 1
        self.obs_buf[env_ids, d] = self.dof_pos[env_ids, 1]

        # elevator
        d += 1
        self.obs_buf[env_ids, d] = self.dof_pos[env_ids, 2]

        # effort
        d += 1
        self.obs_buf[env_ids, d] = self.k_effort_thrust[env_ids]
        d += 1
        self.obs_buf[env_ids, d] = self.k_effort_botthrust[env_ids]
        d += 1
        self.obs_buf[env_ids, d] = self.k_ema_smooth[env_ids, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_ema_smooth[env_ids, 1]

        d += 1
        self.obs_buf[env_ids, d] = self.k_body_areas[env_ids, 0, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_areas[env_ids, 0, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_areas[env_ids, 0, 2]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_areas[env_ids, 1, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_areas[env_ids, 2, 1]

        d += 1
        self.obs_buf[env_ids, d] = self.k_drag_coefs[env_ids, 0, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_drag_coefs[env_ids, 0, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_drag_coefs[env_ids, 0, 2]
        d += 1
        self.obs_buf[env_ids, d] = self.k_drag_coefs[env_ids, 1, 1]

        d += 1
        self.obs_buf[env_ids, d] = self.k_body_torque_coeff[env_ids, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_torque_coeff[env_ids, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_torque_coeff[env_ids, 2]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_torque_coeff[env_ids, 3]
        d += 1
        self.obs_buf[env_ids, d] = self.k_body_torque_coeff[env_ids, 4]

        d += 1
        self.obs_buf[env_ids, d] = self.k_wind[env_ids, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_wind[env_ids, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_wind[env_ids, 2]

        d += 1
        self.obs_buf[env_ids, d] = self.k_wind_mean[env_ids, 0]
        d += 1
        self.obs_buf[env_ids, d] = self.k_wind_mean[env_ids, 1]
        d += 1
        self.obs_buf[env_ids, d] = self.k_wind_mean[env_ids, 2]

        d += 1
        self.obs_buf[env_ids, d] = self.k_blimp_mass[env_ids]
        d += 1
        self.obs_buf[env_ids, d] = self.k_bouyancy[env_ids]

        # pid hint
        for controller in self.controllers:
            a = controller.act(self.obs_buf)

            d += 1
            self.obs_buf[env_ids, d] = a[env_ids, 0]
            d += 1
            self.obs_buf[env_ids, d] = a[env_ids, 1]
            d += 1
            self.obs_buf[env_ids, d] = a[env_ids, 2]
            d += 1
            self.obs_buf[env_ids, d] = a[env_ids, 3]

    def randomize_latent(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        def sample_from_range(some_range, dim=1):
            a, b = some_range
            return (
                torch.rand(size=(len(env_ids), dim), device=self.sim_device) * (b - a)
                + a
            )

        # randomize effort
        self.k_effort_thrust[env_ids] = sample_from_range(self.range_effort_thrust)[0]
        self.k_effort_botthrust[env_ids] = sample_from_range(
            self.range_effort_botthrust
        )[0]
        self.k_ema_smooth[env_ids] = sample_from_range(self.range_ema_smooth, 2)

        # randomize dragbody
        a = sample_from_range(self.range_body_areas0, 3)
        b = sample_from_range(self.range_body_areas1, 3)
        c = sample_from_range(self.range_body_areas2, 3)
        self.k_body_areas[env_ids, 0] = a
        self.k_body_areas[env_ids, 1] = b
        self.k_body_areas[env_ids, 3] = b
        self.k_body_areas[env_ids, 5] = b
        self.k_body_areas[env_ids, 7] = b
        self.k_body_areas[env_ids, 2] = c
        self.k_body_areas[env_ids, 4] = c
        self.k_body_areas[env_ids, 6] = c
        self.k_body_areas[env_ids, 8] = c

        self.k_drag_coefs[env_ids, 0] = sample_from_range(self.range_drag_coefs0, 3) * 2
        self.k_drag_coefs[env_ids, 1:] = (
            sample_from_range(self.range_drag_coefs1, 3)[:, None] * 2
        )

        self.k_body_torque_coeff[env_ids] = sample_from_range(
            self.range_body_torque_coeff, 5
        )

        # randomize wind
        k_wind_mag = sample_from_range(self.range_wind_mag, 3)
        k_wind_dir = (
            2 * torch.rand((len(env_ids), 3), device=self.sim_device) * self.wind_dirs
            - self.wind_dirs
        )
        self.k_wind_mean[env_ids] = k_wind_mag * k_wind_dir
        self.k_wind_std[env_ids] = sample_from_range(self.range_wind_std, 3)

        # randomize bouyancy
        self.k_blimp_mass[env_ids] = sample_from_range(self.range_blimp_mass)[0]
        self.k_bouyancy[env_ids] = torch.normal(
            mean=-self.sim_params.gravity.z * (self.k_blimp_mass[env_ids] - 0.5),
            std=0.1,
        )

    def get_reward(self):
        # retrieve environment observations from buffer
        x = self.obs_buf[:, self.id_relpos[0]]
        y = self.obs_buf[:, self.id_relpos[1]]
        z = self.obs_buf[:, self.id_relpos[2]]

        z_abs = self.obs_buf[:, self.id_absZ]

        angx = self.obs_buf[:, self.id_ang[0]]
        angy = self.obs_buf[:, self.id_ang[1]]

        wx = self.obs_buf[:, self.id_angvel[0]]
        wy = self.obs_buf[:, self.id_angvel[1]]
        wz = self.obs_buf[:, self.id_angvel[2]]

        (
            self.reward_buf[:],
            self.reset_buf[:],
            self.return_buf[:],
            self.truncated_buf[:],
        ) = compute_point_reward(
            x,
            y,
            z,
            z_abs,
            angx,
            angy,
            wx,
            wy,
            wz,
            self.pos_lim,
            self.reset_dist,
            self.reset_buf,
            self.progress_buf,
            self.return_buf,
            self.truncated_buf,
            self.max_episode_length,
        )

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return

        def sampling(size, scale):
            return scale * 2 * (torch.rand(size, device=self.sim_device) - 0.5)

        # randomise initial positions and velocities
        positions = sampling((len(env_ids), self.num_bodies, 3), self.pos_lim)
        positions[..., 2] += self.spawn_height
        positions[..., 2] = torch.where(
            positions[..., 2] <= 2.0, self.spawn_height, positions[..., 2]
        )

        rotations = sampling((len(env_ids), 3), math.pi)
        rotations[..., 0:2] = 0

        # set random pos, rot, vels
        self.rb_pos[env_ids, :] = positions[:]
        self.rb_rot[env_ids, 0, :] = quat_from_euler_xyz(
            rotations[:, 0], rotations[:, 1], rotations[:, 2]
        )

        if self.init_vels:
            self.rb_lvels[env_ids, :] = sampling(
                (len(env_ids), self.num_bodies, 3), self.vel_lim
            )
            self.rb_avels[env_ids, :] = sampling(
                (len(env_ids), self.num_bodies, 3), self.avel_lim
            )

        # sample new waypoint
        self.wp.sample(env_ids)
        self.wp.ang[env_ids, 0:2] = 0
        self.wp.angvel[env_ids, 0:2] = 0

        # domain randomization
        self.randomize_latent(env_ids)

        # clear controller states
        for ctrl in self.controllers:
            ctrl.clear()

        # selectively reset the environments
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.rb_states[::15].contiguous()),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # clear relevant buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.return_buf[env_ids] = 0
        self.truncated_buf[env_ids] = 0

        self.prev_actions[env_ids] = torch.zeros(
            (len(env_ids), self.num_act),
            device=self.sim_device,
            dtype=torch.float,
        )

        # refresh new observation after reset
        self.get_obs()

    def step(self, actions):
        actions = actions.to(self.sim_device).reshape((self.num_envs, self.num_act))
        actions = torch.clamp(actions, -1.0, 1.0)  # [thrust, yaw, stick, pitch]
        self.prev_actions = actions

        # EMA smoothing thrusts
        actions[:, 0] = actions[:, 0] * self.k_ema_smooth[:, 0] + self.prev_actuator[
            :, 0
        ] * (1 - self.k_ema_smooth[:, 0])
        actions[:, 2] = actions[:, 2] * self.k_ema_smooth[:, 1] + self.prev_actuator[
            :, 1
        ] * (1 - self.k_ema_smooth[:, 1])
        bot_thrust = actions[:, 1] * self.k_ema_smooth[:, 0] + self.prev_actuator[
            :, 2
        ] * (1 - self.k_ema_smooth[:, 0])

        self.prev_actuator[:, 0] = actions[:, 0]
        self.prev_actuator[:, 1] = actions[:, 2]
        self.prev_actuator[:, 2] = bot_thrust

        # zeroing out any prev action
        self.actions_tensor[:] = 0.0
        self.torques_tensor[:] = 0.0

        self.actions_tensor[:, 3, 2] = (
            self.k_effort_thrust * (actions[:, 0] + 1) / 2
        )  # propeller
        self.actions_tensor[:, 4, 2] = (
            self.k_effort_thrust * (actions[:, 0] + 1) / 2
        )  # propeller
        self.actions_tensor[:, 7, 1] = (
            self.k_effort_botthrust * bot_thrust
        )  # bot propeller

        # buoyancy
        self.actions_tensor[:] = simulate_buoyancy(
            self.rb_rot, self.k_bouyancy, self.actions_tensor
        )

        # randomize wind
        self.k_wind = torch.normal(mean=self.k_wind_mean, std=self.k_wind_std)

        self.actions_tensor[:], self.torques_tensor[:] = simulate_aerodynamics(
            rb_rot=self.rb_rot,
            rb_avels=self.rb_avels,
            rb_lvels=self.rb_lvels,
            wind=self.k_wind,
            drag_bodies=self.drag_bodies,
            body_areas=self.k_body_areas,
            drag_coefs=self.k_drag_coefs,
            torques_tensor=self.torques_tensor,
            actions_tensor=self.actions_tensor,
            body_torque_coeff=self.k_body_torque_coeff,
        )

        dof_targets = torch.zeros((self.num_envs, 5), device=self.sim_device)
        dof_targets[:, 0] = torch.pi / 2 * actions[:, 2]  # stick
        dof_targets[:, 1] = 0.5 * actions[:, 1]  # bot fin
        dof_targets[:, 4] = -0.5 * actions[:, 1]  # top fin
        dof_targets[:, 2] = 0.5 * actions[:, 3]  # left fin
        dof_targets[:, 3] = -0.5 * actions[:, 3]  # right fin

        # unwrap tensors
        dof_targets = gymtorch.unwrap_tensor(dof_targets)
        forces = gymtorch.unwrap_tensor(self.actions_tensor)
        torques = gymtorch.unwrap_tensor(self.torques_tensor)

        # apply actions
        self.gym.apply_rigid_body_force_tensors(
            self.sim, forces, torques, gymapi.LOCAL_SPACE
        )
        self.gym.set_dof_position_target_tensor(self.sim, dof_targets)

        # simulate and render
        self.simulate()
        if not self.headless:
            self.render()

        # reset environments if required
        self.progress_buf += 1

        self.get_obs()
        self.get_reward()

    def _add_goal_lines(self, num_lines, line_colors, line_vertices, envs):
        num_lines += 1
        line_colors += [[0, 0, 100]]
        for i in range(envs):
            vertices = [
                [
                    self.wp.get_pos_nav()[i, 0].item(),
                    self.wp.get_pos_nav()[i, 1].item(),
                    0,
                ],
                [
                    self.wp.get_pos_nav()[i, 0].item(),
                    self.wp.get_pos_nav()[i, 1].item(),
                    self.wp.get_pos_nav()[i, 2].item(),
                ],
            ]
            if len(line_vertices) > i:
                line_vertices[i] += vertices
            else:
                line_vertices.append(vertices)

        num_lines += 2
        line_colors += [[0, 100, 0], [0, 100, 0]]
        for i in range(envs):
            vertices = [
                [
                    self.wp.pos_hov[i, 0].item(),
                    self.wp.pos_hov[i, 1].item(),
                    0,
                ],
                [
                    self.wp.pos_hov[i, 0].item(),
                    self.wp.pos_hov[i, 1].item(),
                    self.wp.pos_hov[i, 2].item(),
                ],
                [
                    self.wp.pos_hov[i, 0].item(),
                    self.wp.pos_hov[i, 1].item(),
                    self.wp.pos_hov[i, 2].item(),
                ],
                [
                    self.wp.pos_hov[i, 0].item() + math.cos(self.wp.ang[i, 2].item()),
                    self.wp.pos_hov[i, 1].item() + math.sin(self.wp.ang[i, 2].item()),
                    self.wp.pos_hov[i, 2].item(),
                ],
            ]
            if len(line_vertices) > i:
                line_vertices[i] += vertices
            else:
                line_vertices.append(vertices)

        return num_lines, line_colors, line_vertices

    def _add_thrust_lines(self, num_lines, line_colors, line_vertices, envs):
        num_lines += 3
        line_colors += [[200, 0, 0], [200, 0, 0], [200, 0, 0]]

        s = 1
        idx = 3
        roll, pitch, yaw = get_euler_xyz(self.rb_rot[:, idx, :])
        f1, f2, f3 = localToGlobalRot(
            roll,
            pitch,
            yaw,
            self.actions_tensor[:, idx, 0],
            self.actions_tensor[:, idx, 1],
            self.actions_tensor[:, idx, 2],
        )
        roll, pitch, yaw = get_euler_xyz(self.rb_rot[:, 7, :])
        g1, g2, g3 = localToGlobalRot(
            roll,
            pitch,
            yaw,
            self.actions_tensor[:, 7, 0],
            self.actions_tensor[:, 7, 1],
            self.actions_tensor[:, 7, 2],
        )
        for i in range(envs):
            vertices = []
            for idx in [3, 4]:
                vertices.append(
                    [
                        self.rb_pos[i, idx, 0].item(),
                        self.rb_pos[i, idx, 1].item(),
                        self.rb_pos[i, idx, 2].item(),
                    ]
                )
                vertices.append(
                    [
                        self.rb_pos[i, idx, 0].item() - s * f1[i].item(),
                        self.rb_pos[i, idx, 1].item() - s * f2[i].item(),
                        self.rb_pos[i, idx, 2].item() - s * f3[i].item(),
                    ]
                )
            vertices.append(
                [
                    self.rb_pos[i, 7, 0].item(),
                    self.rb_pos[i, 7, 1].item(),
                    self.rb_pos[i, 7, 2].item(),
                ]
            )
            vertices.append(
                [
                    self.rb_pos[i, 7, 0].item() - s * g1[i].item(),
                    self.rb_pos[i, 7, 1].item() - s * g2[i].item(),
                    self.rb_pos[i, 7, 2].item() - s * g3[i].item(),
                ]
            )
            if len(line_vertices):
                line_vertices[i] += vertices
            else:
                line_vertices.append(vertices)
        return num_lines, line_colors, line_vertices

    def _add_drag_lines(self, num_lines, line_colors, line_vertices, envs):
        num_lines += 1
        line_colors += [[200, 0, 0]]

        s = 50
        idx = 12
        roll, pitch, yaw = get_euler_xyz(self.rb_rot[:, idx, :])
        f1, f2, f3 = localToGlobalRot(
            roll,
            pitch,
            yaw,
            self.actions_tensor[:, idx, 0],
            self.actions_tensor[:, idx, 1],
            self.actions_tensor[:, idx, 2],
        )

        for i in range(envs):
            vertices = [
                [
                    self.rb_pos[i, idx, 0].item(),
                    self.rb_pos[i, idx, 1].item(),
                    self.rb_pos[i, idx, 2].item(),
                ],
                [
                    self.rb_pos[i, idx, 0].item() - s * f1[i].item(),
                    self.rb_pos[i, idx, 1].item() - s * f2[i].item(),
                    self.rb_pos[i, idx, 2].item() - s * f3[i].item(),
                ],
            ]
            if len(line_vertices):
                line_vertices[i] += vertices
            else:
                line_vertices.append(vertices)
        return num_lines, line_colors, line_vertices

    def _generate_lines(self):
        num_lines = 0
        # line_colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [200, 0, 0], [0, 200, 0], [0, 0, 200]]
        line_colors = []
        line_vertices = []

        num_lines, line_colors, line_vertices = self._add_goal_lines(
            num_lines, line_colors, line_vertices, self.num_envs
        )
        num_lines, line_colors, line_vertices = self._add_thrust_lines(
            num_lines, line_colors, line_vertices, self.num_envs
        )
        # num_lines, line_colors, line_vertices = self._add_drag_lines(
        #     num_lines, line_colors, line_vertices, self.num_envs
        # )

        return line_vertices, line_colors, num_lines


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def simulate_buoyancy(rb_rot, bouyancy, actions_tensor):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    roll, pitch, yaw = get_euler_xyz(rb_rot[:, 0, :])
    xa, ya, za = globalToLocalRot(
        roll,
        pitch,
        yaw,
        torch.zeros_like(bouyancy),
        torch.zeros_like(bouyancy),
        bouyancy,
    )
    actions_tensor[:, 0, 0] = xa
    actions_tensor[:, 0, 1] = ya
    actions_tensor[:, 0, 2] = za

    return actions_tensor


@torch.jit.script
def simulate_aerodynamics(
    rb_rot,
    rb_avels,
    rb_lvels,
    wind,
    drag_bodies,
    body_areas,
    drag_coefs,
    torques_tensor,
    actions_tensor,
    body_torque_coeff,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    coef = body_torque_coeff[:, 0]  # coef = 0.47
    p = body_torque_coeff[:, 1]  # p = 1.29
    BL4 = body_torque_coeff[:, 2]  # BL4 = 270
    balance_torque = body_torque_coeff[:, 3]  # balance_torque = 5
    fin_torque_coeff = body_torque_coeff[:, 4]  # fin_torque_coeff = 2.5

    D = (1 / 64) * p * coef * BL4
    r, p, y = get_euler_xyz(rb_rot[:, 0, :])
    a, b, c = globalToLocalRot(
        r,
        p,
        y,
        rb_avels[:, 0, 0],
        rb_avels[:, 0, 1],
        rb_avels[:, 0, 2],
    )
    torques_tensor[:, 0, 1] = -D * b * torch.abs(b)
    torques_tensor[:, 0, 2] = -D * c * torch.abs(c)

    r, p, y = get_euler_xyz_multi(rb_rot[:, drag_bodies, :])
    a, b, c = globalToLocalRot(
        r,
        p,
        y,
        rb_lvels[:, drag_bodies, 0] + wind[:, 0:1],
        rb_lvels[:, drag_bodies, 1] + wind[:, 1:2],
        rb_lvels[:, drag_bodies, 2] + wind[:, 2:3],
    )
    # area = body_areas[i]
    aerodynamic_force0 = -drag_coefs[..., 0] * body_areas[..., 0] * a * torch.abs(a)
    aerodynamic_force1 = -drag_coefs[..., 1] * body_areas[..., 1] * b * torch.abs(b)
    aerodynamic_force2 = -drag_coefs[..., 2] * body_areas[..., 2] * c * torch.abs(c)

    actions_tensor[:, drag_bodies, 0] += aerodynamic_force0
    actions_tensor[:, drag_bodies, 1] += aerodynamic_force1
    actions_tensor[:, drag_bodies, 2] += aerodynamic_force2

    # balance pitch torque
    torques_tensor[:, drag_bodies[0], 1] += balance_torque

    # pitch torque
    torques_tensor[:, drag_bodies[0], 1] += fin_torque_coeff * (
        aerodynamic_force1[:, 4] - aerodynamic_force1[:, 6]
    )
    # yaw torque
    torques_tensor[:, drag_bodies[0], 2] += fin_torque_coeff * (
        -aerodynamic_force1[:, 2] + aerodynamic_force1[:, 8]
    )
    return actions_tensor, torques_tensor


@torch.jit.script
def compute_point_reward(
    x_pos,
    y_pos,
    z_pos,
    z_abs,
    angx,
    angy,
    ang_velx,
    ang_vely,
    ang_velz,
    pos_lim,
    reset_dist,
    reset_buf,
    progress_buf,
    return_buf,
    truncated_buf,
    max_episode_length,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor,Tensor, Tensor,float) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    sqr_dist = (x_pos) ** 2 + (y_pos) ** 2 + (z_pos-pos_lim) ** 2

    prox_x_rew_gauss = (torch.exp(-0.01 * sqr_dist) + torch.exp(-0.4 * sqr_dist)) / 2
    # prox_x_rew = torch.where(sqr_dist > 2**2, prox_x_rew_gauss, 1)

    reward = prox_x_rew_gauss

    # adjust reward for reset agents
    reward = torch.where(z_abs < 2, torch.ones_like(reward) * -10.0, reward)
    # reward = torch.where(torch.abs(x_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(torch.abs(y_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(torch.abs(wz) > 45, torch.ones_like(reward) * -200.0, reward)
    # reward = torch.where(x_action < 0, reward - 0.1, reward)
    # reward = torch.where((torch.abs(x_pos) < 0.1) & (torch.abs(y_pos) < 0.1), reward + 1, reward)

    return_buf += reward

    # reset
    reset = torch.where(
        sqr_dist > reset_dist**2, torch.ones_like(reset_buf), reset_buf
    )
    # reset = torch.where(
    #     torch.abs(x_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf
    # )
    # reset = torch.where(
    #     torch.abs(y_pos) > reset_dist, torch.ones_like(reset_buf), reset
    # )
    # reset = torch.where(
    #     z_abs > reset_dist*2, torch.ones_like(reset_buf), reset
    # )
    reset = torch.where(z_abs < 2, torch.ones_like(reset_buf), reset)

    reset = torch.where(
        torch.abs(angx) > torch.pi / 2-0.2, torch.ones_like(reset_buf), reset
    )
    reset = torch.where(
        torch.abs(angy) > torch.pi / 2-0.2, torch.ones_like(reset_buf), reset
    )

    reset = torch.where(
        torch.abs(ang_velx) > torch.pi / 2, torch.ones_like(reset_buf), reset
    )
    reset = torch.where(
        torch.abs(ang_vely) > torch.pi / 2, torch.ones_like(reset_buf), reset
    )
    reset = torch.where(
        torch.abs(ang_velz) > torch.pi / 2, torch.ones_like(reset_buf), reset
    )

    truncated_buf = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        truncated_buf,
    )

    return reward, reset, return_buf, truncated_buf
