
import cma
from utils.chamfer.chamfer import chamfer_distance
from utils.utils import *

def get_cost_function(target, parts, n_part, dof):
    if not torch.is_tensor(target):
        target = torch.tensor(target).cuda().float()
    if not torch.is_tensor(parts):
        parts = torch.tensor(parts).cuda().float()
    def cd(poses):
        if not torch.is_tensor(poses):
            poses = torch.tensor(poses).cuda().float()
        poses = poses.reshape(n_part, -1)
        rotation_rep = 'quat' if dof==7 else '6D'
        parts_pred = apply_pose(parts, poses, rotation_rep)
        target_pred = parts_pred.reshape(-1, 3)
        cd = chamfer_distance(target, target_pred)
        cd_mean = cd.mean().detach().cpu().item()
        return cd_mean
    fn = lambda poses: cd(poses)
    return fn

def optimize_pose(target, parts, n_part, poses=None, timeout=5):
    opts = cma.CMAOptions()
    opts.set('timeout', timeout)
    if poses is None:
        poses = np.array([[0,0,0,1,0,0,0]]*n_part)
    else:
        poses = poses.detach().cpu().numpy()
    poses_unroll = poses.reshape(-1)
    es = cma.CMAEvolutionStrategy(poses_unroll, 0.1, opts)
    fn = get_cost_function(target, parts, n_part, 7)
    es.optimize(fn)
    poses_optimize = es.result[0].reshape(n_part, 7)
    poses_optimize = torch.tensor(poses_optimize).to(target.device).float()
    return poses_optimize