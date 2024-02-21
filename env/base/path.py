import torch


class Bezier:
    def TwoPoints(t, P1, P2):
        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        newpoints = []
        for i1 in range(0, len(points) - 1):
            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
        return newpoints

    def Point(t, points):
        newpoints = points
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
        return newpoints[0]


class Linear:
    def __init__(self, kWayPt) -> None:
        self.kWayPt = kWayPt

    def update_vel(self, wppos, rbpos, idx, Kv=0.5):
        n_env = wppos.shape[0]

        cur_pos = wppos[range(n_env), self.check_idx(idx).squeeze()]  # [N, 3]
        prev_pos = wppos[range(n_env), self.check_idx(idx - 1).squeeze()]  # [N, 3]
        path = cur_pos - prev_pos

        k = torch.einsum("ij,ij->i", rbpos - prev_pos, path) / torch.einsum(
            "ij,ij->i", path, path
        )
        k = torch.where(k > 1, 1 - k, k)
        desired_vel = Kv * (path + prev_pos + k[:, None] * path - rbpos)
        return desired_vel

    def check_idx(self, idx):
        idx = torch.where(idx > self.kWayPt - 1, 0, idx)
        idx = torch.where(idx < 0, self.kWayPt - 1, idx)
        return idx
