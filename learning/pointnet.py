
import torch
import torch.nn as nn
import torch.functional as F

class Conv1dBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(Conv1dBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True))


class FCBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(FCBNReLU, self).__init__(
            nn.Linear(in_planes, out_planes, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True))


class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        self.encoder = nn.Sequential(
            Conv1dBNReLU(3, 64),
            Conv1dBNReLU(64, 128),
            Conv1dBNReLU(128, 256))
        self.decoder = nn.Sequential(
            FCBNReLU(256, 128),
            FCBNReLU(128, 64),
            nn.Linear(64, 6))

    @staticmethod
    def f2R(f):
        r1 = F.normalize(f[:, :3])
        proj = (r1.unsqueeze(1) @ f[:, 3:].unsqueeze(2)).squeeze(2)
        r2 = F.normalize(f[:, 3:] - proj * r1)
        r3 = r1.cross(r2)
        return torch.stack([r1, r2, r3], dim=2)

    def forward(self, pts):
        f = self.encoder(pts)
        f, _ = f.max(dim=2)
        f = self.decoder(f)
        R = self.f2R(f)
        return R @ pts


class PointNet(nn.Module):
    def __init__(self, out_dim):
        super(PointNet, self).__init__()
        d_input = 3
        self.encoder = nn.Sequential(
            Conv1dBNReLU(d_input, 64),
            Conv1dBNReLU(64, 128),
            Conv1dBNReLU(128, 256))
        self.decoder = nn.Sequential(
            Conv1dBNReLU(512, 256),
            Conv1dBNReLU(256, 128),
            Conv1dBNReLU(128, 64),
            nn.Conv1d(64, out_dim, kernel_size=1))

    def forward(self, pts):
        f_loc = self.encoder(pts.transpose(1, 2))
        f_glob, _ = f_loc.max(dim=2)
        f_glob = f_glob.unsqueeze(2).expand_as(f_loc)
        y = self.decoder(torch.cat([f_loc, f_glob], dim=1))
        return y.transpose(1, 2)