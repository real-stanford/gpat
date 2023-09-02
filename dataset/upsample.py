
import trimesh
import numpy as np
from utils.utils import *
import torch
from tqdm import tqdm

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces)
    v = np.vstack(vertices)
    return v, f

def sample_pc(v, f, n_points=1000):
    mesh = trimesh.Trimesh(vertices=v, faces=f-1)
    points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
    return points

def upsample_pcd(fn):
    obj_file = fn / 'target.obj'
    if not obj_file.exists() or (fn / 'target_100k.npy').exists():
        return
    
    v, f = load_obj(obj_file)
    pcd = sample_pc(v, f, n_points=100000)
    tar_pc = np.load(fn / 'target.npy')
    pcd_torch = torch.tensor(pcd, requires_grad=False).cuda()
    tar_pc_torch = torch.tensor(tar_pc, requires_grad=False).cuda()
    closest_pt = (pcd_torch.unsqueeze(1) - tar_pc_torch.unsqueeze(0)).norm(dim=2).argmin(dim=1)
    closest_pt = closest_pt.detach().cpu().numpy()
    np.save(fn / 'target_100k.npy', pcd)
    np.save(fn / 'target_100k_label.npy', closest_pt)

train_root = wrap_path('dataset/partnet/train')
val_root = wrap_path('dataset/partnet/val')
test_root = wrap_path('dataset/partnet/test')
cats = ['Bed', 'Dishwasher', 'Display', 'Door', 'Faucet', 'Microwave', 'Refrigerator', 'StorageFurniture', 'TrashCan', 'Vase', 'Chair', 'Table', 'Lamp']
# cats = ['Chair', 'Table', 'Lamp']
dirs = [f'{train_root}/{cat}' for cat in cats]
dirs += [f'{val_root}/{cat}' for cat in cats]
dirs += [f'{test_root}/{cat}' for cat in cats]

torch.cuda.set_device(0)

for root_folder in dirs:
    print(root_folder)
    root_folder = wrap_path(root_folder)
    for fn in tqdm(list(root_folder.iterdir())):
        upsample_pcd(fn)
