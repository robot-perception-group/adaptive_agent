import torch
import math
import numpy as np


def PointsInCircum(r,z,n=8):
    pi = math.pi
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r, z) for x in range(0,n+1)]

def PointsInSquare(r,z,n=8):
    pi = math.pi
    return [(math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r, z) for x in range(0,n+1)]

class FixWayPoints:
    """fixed waypoints"""

    def __init__(
        self,
        device,
        num_envs,
        pos_lim=None,
        trigger_dist=2,
        style="hourglass",
        vel_update_freq=4,
        **kwargs,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.trigger_dist = trigger_dist

        self.pos_lim = pos_lim
        
        self.kWayPt = 8
        if style=="square":
            wps = torch.tensor(
                [[[15, -15, 20], [15, 0, 20], [15, 15, 20], [0, 15, 20], [-15, 15, 20], [-15, 0, 20], [-15, -15, 20], [0, -15, 20]]],
                device=self.device,
                dtype=torch.float32,
            )
        elif style=="hourglass":
            wps = torch.tensor(
                [[[15, 15, 20], [0, 15, 20], [-15, 15, 20], [0, 0, 20], [15, -15, 20], [0, -15, 20], [-15, -15, 20], [0, 0, 20]]],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            wps = torch.tensor(
                PointsInCircum(15, 20, self.kWayPt-1),
                device=self.device,
                dtype=torch.float32,
            )
        self.pos_nav = torch.tile(wps, (self.num_envs, 1, 1))

        self.pos_hov = torch.tile(
            torch.tensor([0, 0, 20], device=self.device, dtype=torch.float32),
            (self.num_envs, 1),
        )

        self.vel = torch.tile(
            torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32),
            (self.num_envs, 1),
        )
        self.velnorm = torch.tile(
            torch.tensor([3.0], device=self.device, dtype=torch.float32),
            (self.num_envs, 1),
        )

        self.ang = torch.tile(
            torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32),
            (self.num_envs, 1),
        )

        self.angvel = torch.tile(
            torch.tensor([0, 0, 0], device=self.device, dtype=torch.float32),
            (self.num_envs, 1),
        )

        self.idx = torch.randint(0, self.kWayPt, size=(self.num_envs, 1), device=self.device).to(torch.long)
        self.vel_update_freq = vel_update_freq
        self.cnt = 0

    def sample(self, env_ids):
        pass

    def get_pos_nav(self, idx=None):
        if idx is not None:
            return self.pos_nav[range(self.num_envs), self.check_idx(idx).squeeze()]
        else:
            return self.pos_nav[range(self.num_envs), self.check_idx(self.idx).squeeze()]


    def update_state(self, rb_pos):
        """check if robot is close to waypoint"""
        dist = torch.norm(
            rb_pos[:, 0:2] - self.get_pos_nav(self.idx)[:, 0:2],
            dim=1,
            keepdim=True,
        )
        trigger = torch.where(dist <= self.trigger_dist, 1.0, 0.0)

        self.idx = torch.where(dist <= self.trigger_dist, self.idx + 1, self.idx)
        self.idx = self.check_idx(self.idx)

        self.cnt += 1
        if self.cnt%self.vel_update_freq==0:
            self.update_vel(rb_pos, Kv=self.velnorm*2)
        return trigger

    def reset(self, env_ids):
        # self.idx[env_ids] = 0
        pass

    def update_vel(self, rbpos, Kv=2):
        cur_pos = self.get_pos_nav(self.idx)
        target_vec = cur_pos - rbpos
        target_vec = target_vec / (torch.norm(target_vec, dim=1, keepdim=True)+1e-6)
        self.vel = Kv*target_vec

    def check_idx(self, idx):
        idx = torch.where(idx > self.kWayPt - 1, 0, idx)
        idx = torch.where(idx < 0, self.kWayPt - 1, idx)
        return idx


class RandomWayPoints(FixWayPoints):
    """a random generated goal during training"""

    def __init__(
        self,
        device,
        num_envs,
        init_pos=None,
        init_vel=None,
        init_velnorm=None,
        init_ang=None,
        init_angvel=None,
        rand_pos=True,
        rand_ang=True,
        rand_vel=True,
        rand_angvel=True,
        pos_lim=None,
        vel_lim=None,
        angvel_lim=None,
        kWayPt=1,
        wp_dist=10 / np.sqrt(3),  # [m] generate next wp within range
        trigger_dist=2,  # [m] activate next wp if robot within range
        min_z=5,
        max_z=40,
        reset_dist=30,
    ) -> None:
        super().__init__(
            device=device,
            num_envs=num_envs,
            pos_lim=pos_lim,
            trigger_dist=trigger_dist,
        )

        self.num_envs = num_envs
        self.device = device
        self.kWayPt = kWayPt

        self.pos_lim = pos_lim
        self.vel_lim = vel_lim
        self.angvel_lim = angvel_lim

        self.rand_pos = rand_pos
        self.rand_ang = rand_ang
        self.rand_vel = rand_vel
        self.rand_angvel = rand_angvel

        self.wp_dist = wp_dist
        self.trigger_dist = torch.tensor(trigger_dist).to(self.device)
        self.min_z = min_z
        self.max_z = max_z
        self.reset_dist = reset_dist

        self.idx = torch.zeros(self.num_envs, 1, device=self.device).to(torch.long)

        assert self.kWayPt > 1, "number of waypoints less than 1"
        wps = torch.tensor(
            PointsInCircum(self.pos_lim/2, self.pos_lim, self.kWayPt-1),
            device=self.device,
            dtype=torch.float32,
        )
        self.pos_nav = torch.tile(wps, (self.num_envs, 1, 1))

        if init_pos is not None:
            self.init_pos = init_pos
            self.pos_hov = torch.tile(
                torch.tensor(init_pos, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )

        if init_vel is not None:
            self.init_vel = init_vel
            self.vel = torch.tile(
                torch.tensor(init_vel, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )

        if init_velnorm is not None:
            self.init_velnorm = init_velnorm
            self.velnorm = torch.tile(
                torch.tensor(init_velnorm, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )

        if init_ang is not None:
            self.init_ang = init_ang
            self.ang = torch.tile(
                torch.tensor(init_ang, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )

        if init_angvel is not None:
            self.init_angvel = init_angvel
            self.angvel = torch.tile(
                torch.tensor(init_angvel, device=self.device, dtype=torch.float32),
                (self.num_envs, 1),
            )

        self.idx = torch.zeros(self.num_envs, 1, device=self.device).to(torch.long)

    def sample(self, env_ids):
        if self.rand_pos:
            self.pos_nav[env_ids] = self._sample_nav_goals(
                len(env_ids), self.pos_lim, kWP=self.kWayPt, dist=self.wp_dist, min_dist=self.trigger_dist+1
            )

        if self.rand_vel:
            self.velnorm[env_ids] = torch.abs(
                self._sample((len(env_ids), 1), self.vel_lim-1)
            ) + 1 # min 1 [m/s]

        if self.rand_ang:
            self.ang[env_ids] = self._sample((len(env_ids), 3), math.pi)

        if self.rand_angvel:
            self.angvel[env_ids] = self._sample((len(env_ids), 3), self.angvel_lim)

    def _sample(self, size, scale=1):
        return scale * 2 * (torch.rand(size, device=self.device) - 0.5)

    def _sample_nav_goals(self, size, scale, kWP, dist, min_dist):
        """next wp is spawned [dist] from prev wp"""
        pos = torch.zeros((size, kWP, 3), device=self.device)
        invalid = torch.ones(size, 1, dtype=torch.bool, device=self.device)
        while invalid.any():
            pos[:,0] = torch.where(invalid, self._sample((size, 3), scale), pos[:,0])
            invalid = self._pos_valid(pos[:, 0])

        for i in range(kWP - 1):
            pos[:, i + 1] = self._sample_on_dist(pos[:, i], dist, min_dist)

        return pos

    def _sample_on_dist(self, pos, dist, min_dist):
        invalid = torch.ones(pos.shape[0], 1, dtype=torch.bool, device=self.device)
        newpos = torch.zeros_like(pos, device=self.device)

        while invalid.any():
            x = self._sample(pos.shape)
            x = dist * x / torch.norm(x, dim=1, keepdim=True)
            newpos = torch.where(invalid, pos + x, newpos)
            invalid = self._pos_valid(newpos)
            invalid = self._planardist_greater_than_mindist(pos, newpos, min_dist, invalid)
        
        return newpos

    def _pos_valid(self, pos):
        invalid = torch.zeros(pos.shape[0], 1, dtype=torch.bool, device=self.device)
        
        pos = pos.unsqueeze(2)
        invalid = torch.where(torch.abs(pos[:, 0]) >= self.pos_lim, torch.ones_like(invalid), invalid)
        invalid = torch.where(torch.abs(pos[:, 1]) >= self.pos_lim, torch.ones_like(invalid), invalid)
        invalid = torch.where(pos[:, 2] >= self.max_z, torch.ones_like(invalid), invalid)
        invalid = torch.where(pos[:, 2] <= self.min_z, torch.ones_like(invalid), invalid)
        return invalid
    
    def _planardist_greater_than_mindist(self, pos, newpos, min_dist, invalid):
        if invalid is None:
            invalid = torch.zeros(pos.shape[0], 1, dtype=torch.bool, device=self.device)

        planar_dist = torch.norm(pos[:, 0:2]-newpos[:, 0:2], dim=1, keepdim=True)
        invalid = torch.where(planar_dist <= min_dist, torch.ones_like(invalid), invalid)
        return invalid

    def reset(self, env_ids):
        self.idx[env_ids] = 0
        self.sample(env_ids)
