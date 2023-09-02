
import numpy as np
import random
import torch
from pathlib import Path
import dominate
import matplotlib
import matplotlib.cm as cm
import pytorch3d.transforms as tf
import open3d as o3d
import struct
from sklearn.decomposition import PCA
import trimesh
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import scipy
from scipy.spatial.transform import Rotation
from datetime import datetime

def evaluate(device, model, dataset_class, args, visualizer=None, html_writer=None):

    if args.nviz != 0:
        args.results_dir.mkdir(exist_ok=True, parents=True)
    test_dirs = load_data(args.eval_dir)
    if args.neval > 0:
        test_dirs = test_dirs[:3*args.neval]
    testset = dataset_class(test_dirs, device=device, args=args)

    collate_fn = getattr(dataset_class, "collate", None)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    total_sample, total_viz = 0, 0
    loss_acc, metric_acc, info_acc = 0, {}, {}
    
    for datapoint in tqdm(testloader):
        if args.neval != -1 and total_sample >= args.neval: break
        if datapoint is None: continue
        batch_size = len(datapoint[0])
        with torch.no_grad():
            loss, preds, metric, info, log = model.forward(datapoint, is_train=False)
        loss_acc += loss * batch_size
        for k, v in metric.items():
            if k in metric_acc.keys(): metric_acc[k] += v * batch_size
            else: metric_acc[k] = v * batch_size
        for k, v in info.items():
            if k in info_acc.keys(): info_acc[k] += v
            else: info_acc[k] = v
        total_sample += batch_size
        if visualizer is not None:
            for b in range(batch_size):
                if args.nviz != -1 and total_viz >= args.nviz: break
                visualizer(args.results_dir, total_viz, 
                        [item[b] for item in datapoint], preds[b], 
                        {key: val[b] for key, val in log.items()})
                total_viz += 1
    loss_acc /= total_sample
    for k in metric_acc.keys():
        metric_acc[k] /= total_sample
    for k, v in info_acc.items():
        info_acc[k] = np.array(info_acc[k])
    if total_viz != 0 and html_writer is not None:
        html_writer(args.results_dir, total_viz, *info_acc.values(), name=args.exp)
    
    metric_tqdm = {'loss': loss}
    for k, v in metric_acc.items():
        metric_tqdm[f'{k}'] = v
    print_mesg = f'{args.exp} \t'
    for k, v in metric_tqdm.items():
        print_mesg += f'{k}: {v:.5f} '
    print(print_mesg)

def train(device, model, dataset_class, args):
    args = deepcopy(args)
    args_eval = deepcopy(args)
    args_eval.eval=True
    
    print(f'Experiment: {args.exp}')
    args.logs_dir.mkdir(exist_ok=True, parents=True)
    logger = SummaryWriter(args.logs_dir)

    train_dirs = load_data(args.train_data_root)
    test_dirs = load_data(args.test_data_root)

    trainset = dataset_class(train_dirs, device=device, args=args)
    testset = dataset_class(test_dirs, device=device, args=args_eval)
    print(f'Train {len(trainset)}; Test {len(testset)}')

    collate_fn = getattr(dataset_class, "collate", None)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    start_epoch = 0
    pbar = tqdm(range(start_epoch, start_epoch+args.nepoch))
    for epoch in pbar:
        loss_train, metric_train = run_epoch(model, trainloader, is_train=True)
        metric_tqdm = {'loss': loss_train,
                       'epoch': epoch}
        logger.add_scalar(f'loss', loss_train, epoch)
        for k, v in metric_train.items():
            logger.add_scalar(f'{k}', v, epoch)
            metric_tqdm[f'{k}'] = v
        
        loss_test, metric_test = run_epoch(model, testloader, is_train=False)
        logger.add_scalar(f'loss_test', loss_test, epoch)
        metric_tqdm['loss_t'] = loss_test
        for k, v in metric_test.items():
            logger.add_scalar(f'{k}_t', v, epoch)
            metric_tqdm[f'{k}_t'] = v
        
        if (epoch+1) % args.interval == 0:
            model.save(epoch+1)
        pbar.set_postfix(metric_tqdm)

def load_data(root, shuffle=False):
    if isinstance(root, Path):
        root = wrap_path(root)
        dirs = list(root.iterdir())
    elif type(root) == list:
        dirs = []
        for r in root:
            r = wrap_path(r)
            dirs += list(r.iterdir())
    dirs = sorted(dirs)
    if shuffle: random.shuffle(dirs)
    return dirs

def run_epoch(model, dataloader, is_train):
    total_sample = 0
    loss_acc = 0
    metric_acc = {}
    for datapoint in dataloader:
        if datapoint is None: continue
        batch_size = len(datapoint[0])
        total_sample += batch_size
        if is_train:
            loss, _, metric, _, _ = model.forward(datapoint, is_train=is_train)
        else:
            with torch.no_grad():
                loss, _, metric, _, _ = model.forward(datapoint, is_train=is_train)
        loss_acc += loss * batch_size
        for k, v in metric.items():
            if k in metric_acc.keys():
                metric_acc[k] += v * batch_size
            else:
                metric_acc[k] = v * batch_size
    loss_acc /= total_sample
    for k in metric_acc.keys():
        metric_acc[k] /= total_sample
    return loss_acc, metric_acc

def prune_pc(points):
    center = np.mean(points, axis=0, keepdims=True)
    dist_from_center = np.linalg.norm(points - center, axis=1)
    mean_dist = np.mean(dist_from_center)
    std_dist = np.std(dist_from_center)
    inds = dist_from_center-mean_dist < 1*std_dist
    return points[inds, :]

def bbox(points):
    try: 
        to_origin, size = trimesh.bounds.oriented_bounds(obj=points, angle_digits=1)
        center = to_origin[:3, :3].transpose().dot(-to_origin[:3, 3])
    
        xdir = to_origin[0, :3]
        ydir = to_origin[1, :3]
        zdir = to_origin[2, :3]
    except:
        points = np.array(points)
        pca = PCA()
        pca.fit(points)
        pcomps = pca.components_
        points_local = np.matmul(pcomps, points.transpose()).transpose()
        all_max = points_local.max(axis=0)
        all_min = points_local.min(axis=0)
        center = np.dot(np.linalg.inv(pcomps), (all_max + all_min) / 2)
        size = all_max - all_min
        xdir = pcomps[0, :]
        ydir = pcomps[1, :]
        zdir = np.cross(xdir, ydir)
        xdir /= np.linalg.norm(xdir)
        ydir /= np.linalg.norm(ydir)
        zdir /= np.linalg.norm(zdir)

    R = np.vstack([xdir, ydir, zdir]).transpose().astype(np.float32)
    t = center.astype(np.float32)
    
    return R, t, size

def viz_pointcloud(save_path, pts, clrs=None, normals=None):
    pts = pts.copy().astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if clrs is not None:
        clrs = clrs.copy().astype(np.float64)/255
        pcd.colors = o3d.utility.Vector3dVector(clrs)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(save_path))
    vis.destroy_window()

def write_pointcloud(filename, xyz_points, rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,\
        'Input RGB colors should be Nx3 array with values 0-255 and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tobytes(),rgb_points[i,1].tobytes(),
                                        rgb_points[i,2].tobytes())))
    fid.close()

def apply_pose(pcs, poses, rotation='quat', return_rotated=False):
    """Apply poses to pointclouds in batch

    Args:
        pcs (..., N, 3): point clouds
        poses (..., 7): translation + quaternion (scalar first)

    Returns:
        (..., N, 3): transformed pointclouds
    """
    N = pcs.shape[-2]
    t, q = poses[...,:3], poses[...,3:]    
    if rotation == 'quat':
        R = tf.quaternion_to_matrix(q).reshape(-1, 3, 3)
    elif rotation == '6D':
        R = tf.rotation_6d_to_matrix(q).reshape(-1, 3, 3)
    t = t.reshape(-1, 3).unsqueeze(-1)
    pcs_reshape = pcs.reshape(-1, N, 3).transpose(-1,-2)
    pcs_rot = torch.bmm(R, pcs_reshape)
    pcs_trans = pcs_rot+t
    pcs_rot = pcs_rot.transpose(-1,-2).reshape(pcs.shape)
    pcs_trans = pcs_trans.transpose(-1,-2).reshape(pcs.shape)
    if return_rotated:
        return pcs_trans, pcs_rot
    return pcs_trans
    
def get_transformation_matrix(pose):
    t, q = pose[...,:3], pose[...,3:]
    R = tf.quaternion_to_matrix(q) # (..., 3, 3)
    T = torch.zeros(*R.shape[:-2], 4, 4).to(pose.device)
    T[..., 3, 3] = 1
    T[...,:3, :3] = R
    T[...,:3, 3] = t
    return T

def get_translation_quaternion(T):
    R = T[...,:3, :3]
    t = T[...,:3, 3]
    quat = tf.matrix_to_quaternion(R)
    pose = torch.cat([t, quat], dim=-1)
    return pose
    
def mult_pose(pose1, pose2):
    """multiply two poses, pose1 applied to pose2

    Args:
        pose1: (...,7), translation + quaternion with real part frist
        pose2: (...,7), translation + quaternion with real part frist
    """
    T1 = get_transformation_matrix(pose1) # (..., 4, 4)
    T2 = get_transformation_matrix(pose2)
    T = torch.bmm(T1.reshape(-1, 4, 4), T2.reshape(-1, 4, 4)).reshape(T1.shape)
    pose = get_translation_quaternion(T)
    return pose

# Credits: Zhenjia Xu <xuzhenjia@cs.columbia.edu>, https://github.com/columbia-robovision/html-visualization
def html_visualize(web_path, data, ids, cols, title='visualization', html_file_name:str="index.html"):
    """Visualization in html.
    
    Args:
        web_path: string; directory to save webpage. It will clear the old data!
        data: dict; 
            key: {id}_{col}. 
            value: Path / str / list of str
                - Path: Figure .png or .gif path
                - text: string or [string]
        ids: [string]; name of each row
        cols: [string]; name of each column
        title: (optional) string; title of the webpage (default 'visualization')
    """
    web_path = Path(web_path)
    web_path.parent.mkdir(parents=True, exist_ok=True)

    with dominate.document(title=title) as web:
        dominate.tags.h1(title)
        with dominate.tags.table(border=1, style='table-layout: fixed;'):
            with dominate.tags.tr():
                with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', width='64px'):
                    dominate.tags.p('id')
                for col in cols:
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', ):
                        dominate.tags.p(col)
            for id in ids:
                with dominate.tags.tr():
                    bgcolor = 'F1C073' if id.startswith('train') else 'C5F173'
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', bgcolor=bgcolor):
                        for part in id.split('_'):
                            dominate.tags.p(part)
                    for col in cols:
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='top'):
                            key = f'{id}_{col}'
                            if key in data:
                                value = data[key]
                            else:
                                value = ""
                            if isinstance(value, str) and (value.endswith(".gif") or value.endswith(".png") or value.endswith(".jpg")):
                                dominate.tags.img(style='height:128px', src=data[f"{id}_{col}"])
                            elif isinstance(value, list) and isinstance(value[0], str):
                                for v in value:
                                    dominate.tags.p(v)
                            else:
                                dominate.tags.p(str(value))

    with open(web_path / html_file_name, "w") as fp:
        fp.write(web.render())

PALETTE = np.array(
    [
        [0,0,0],
        [255, 87, 89],  # red
        [89, 169, 79],  # green
        [78, 121, 167],  # blue
        [156, 117, 95],  # brown
        [176, 122, 161],  # purple
        [118, 183, 178],  # cyan
        [237, 201, 72],  # yellow
        [186, 176, 172],  # gray
        [242, 142, 43],  # orange
        [255, 157, 167],  # pink
        [153, 0, 153],  # pink
    ]
)

def scalar2rgb(scalar_values):
    """
    convert an array of grayscale values to RGB values
    Todo: revert this! make this more beautiful
    """
    scalar_values = np.array(scalar_values).astype(float)
    rgb_vals = np.zeros((scalar_values.shape[0], 3))
    for i in range(11):
        rgb_vals[scalar_values==i] = PALETTE[i]
    return rgb_vals.astype(np.uint8)

def grayscale2rgb(grayscale_values, normalize=True):
    """
    convert an array of grayscale values to RGB values
    Todo: revert this! make this more beautiful
    """
    if normalize:
        norm = matplotlib.colors.Normalize(vmin=np.min(grayscale_values), vmax=np.max(grayscale_values), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='jet')
    else:
        mapper = cm.ScalarMappable(cmap='jet')
    return mapper.to_rgba(grayscale_values)[:,:3] * 255

def wrap_path(path):
    """create a pathlib.Path object"""
    if type(path) == str:
        return Path(path)
    elif isinstance(path, Path):
        return path
    print(f"Error: {path} not a str or path.")
    return None

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed