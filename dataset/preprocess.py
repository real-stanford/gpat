'''prepare data 

    target.npy: (5000, 3), target point-cloud, required.
    parts.npy: (K, 1000, 3), parts point-clouds, required.
    target_100k.npy # (100000, 3), a dense target point-cloud. Optional.
    target_100k_labels.npy # (10000), indicates the nearest neighbor of target_100k.npy in target.npy, with values from [0, 5000). Optional.
    poses.npy # (K, 7), GT poses for each part. First three coordinates denote (x, y, z) position, last four coordinates denote a quaternion with real-part first. Optional.
    labels.npy: (5000), GT segmentation label of the target, each index takes a value from [0, K). Optional.
    eq_class.npy: (K), equivalence classes of the parts. For example, [0,0,1,2,2,2] means that the first two parts are equivalent, and last three parts are equivalent.  Optional.
'''

import json
from utils.utils import *

def export_obj(out, v, f):
    with open(out, 'w') as fout:
        for i in range(v.shape[0]):
            fout.write('v %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2]))
        for i in range(f.shape[0]):
            fout.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

def get_shape_info(root_to_data_folder, index, lev):
    # get children
    fn = root_to_data_folder / f"{index}/result_after_merging.json"
    root_to_load_file = []
    with open( fn, "r" ) as f:
        root_to_load_file = json.load(f)[0]
    parts_objs, parts_names = get_parts_objs(root_to_load_file, lev, '')
    if len(parts_objs) > 20: return None
    parts_v = []
    parts_f = []
    for part_objs in parts_objs:
        obj_fns = [root_to_data_folder / f'{str(index)}/objs/{obj}.obj' for obj in part_objs]
        vs=[]
        fs=[]
        for obj_fn in obj_fns:
            v,f = load_obj(obj_fn)
            vs.append(v)
            fs.append(f)
        v,f = merge_objs(vs,fs)
        parts_v.append(v)
        parts_f.append(f)
    parts_points, gt_part_points, tar_pc, tar_100k, tar_mesh, Rs, ts, sizes = get_norm_parts_points(parts_v, parts_f)
    
    return parts_points, gt_part_points, tar_pc, tar_100k, tar_mesh, Rs, ts, parts_names, sizes

def get_parts_objs(root_to_load_file, lev, root_to_load_file_name):
    try:
        children = root_to_load_file['children']
    except KeyError:
        return [root_to_load_file['objs']], [root_to_load_file_name + '/' + root_to_load_file['name']]
    parts_objs = []
    parts_names = []
    for child in children:
        if root_to_load_file_name + '/' + root_to_load_file['name'] + '/' + child['name'] in lev:
            parts_objs.append(child['objs'])
            parts_names.append(root_to_load_file_name + '/' + root_to_load_file['name'] + '/' + child['name'])
        else:
            child_objs, child_names = get_parts_objs(child, lev, root_to_load_file_name + '/' + root_to_load_file['name'])
            parts_objs = parts_objs + child_objs
            parts_names = parts_names + child_names 
    return parts_objs, parts_names


def get_upsampled_pc_label(upsampled_pcd, pcd):    
    upsampled_pcd_torch = torch.tensor(upsampled_pcd, requires_grad=False).cuda()
    pcd_torch = torch.tensor(pcd, requires_grad=False).cuda()
    closest_pt = (upsampled_pcd_torch.unsqueeze(1) - pcd_torch.unsqueeze(0)).norm(dim=2).argmin(dim=1)
    closest_pt_np = closest_pt.detach().cpu().numpy()
    return closest_pt_np

def get_norm_parts_points(parts_v,parts_f):
    
    max_size = 0
    gt_part_points=[]
    parts_points=[]
    Rs = []
    ts = []
    sizes = []
    for v,f in zip(parts_v,parts_f):
        points = sample_pc(v,f)
        gt_part_points.append(points)
        R, t, size = bbox(points)
        Rs.append(R)
        ts.append(t)
        sizes.append(size)
        points_tf = np.dot(points -t, np.linalg.inv(R).transpose()) 
        parts_points.append(points_tf)
        if max_size < np.max(size): max_size = np.max(size)
    parts_points = np.array(parts_points) / max_size
    gt_part_points = np.array(gt_part_points) / max_size
    
    ts = np.array(ts) / max_size
    Rs = np.array(Rs)
    sizes = np.array(sizes)
    
    all_v, all_f = merge_objs(parts_v, parts_f)
    all_v = all_v / max_size
    tar_mesh = (all_v, all_f)
    tar_pc = sample_pc(all_v,all_f,n_points=5000)
    tar_100k = sample_pc(all_v,all_f,n_points=100000)
    
    return parts_points, gt_part_points, tar_pc, tar_100k, tar_mesh, Rs, ts, sizes

def merge_objs(vs,fs):
    
    newv = []
    newf = []
    num = 0

    for i in range(len(vs)):
        fs[i] += num
        num += len(vs[i])
    
    newv = np.concatenate(vs,axis=0)
    newf = np.concatenate(fs,axis=0)
    return newv,newf

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

def is_eq(size1,size2):

    mean_bs = (np.linalg.norm(size1) + np.linalg.norm(size2)) / 2
    bs_diff = np.linalg.norm(size1 - size2)
    error = bs_diff / mean_bs
    
    box_thre = 0.1
    if error < box_thre:
        return True
    return False

def get_eq_classes(part_sizes,part_ids):
    class_ids = list(set(part_ids))
    geo_ind = 0
    eq_classes = [0 for _ in range(len(part_ids))]
    for class_ind in class_ids:

        list_of_this_class = []
        for ind,part_id in enumerate(part_ids):
            if part_id == class_ind:
                list_of_this_class.append(ind)
        for ind,part_ind in enumerate(list_of_this_class):
            flag = 0
            for pre_part_ind in list_of_this_class[:ind]:
                if is_eq(part_sizes[pre_part_ind],part_sizes[part_ind]):
                    flag = 1
                    eq_classes[part_ind] = eq_classes[pre_part_ind]
                    break
            if flag == 0:
                eq_classes[part_ind] = geo_ind
                geo_ind += 1
    return eq_classes

def get_seg_labels(tar_pc, part_pcs):
    tar_pc = torch.tensor(tar_pc, requires_grad=False).cuda()
    part_pcs = torch.tensor(part_pcs, requires_grad=False).cuda()
    label = (tar_pc.unsqueeze(0).unsqueeze(0) - part_pcs.unsqueeze(2)).norm(dim=3).min(dim=1)[0]
    label = label.argmin(dim=0).detach().cpu().numpy()
    return label

def get_blocks(type):
    block_pcs = np.load(f'dataset/non_exact_parts/{type}.npy')
    block_bbox = []
    for block_pc in block_pcs:
        R_b, t_b, size_b = bbox(block_pc)
        block_bbox.append([R_b, t_b, size_b])
    device = torch.device('cuda')
    block_pcs_torch = torch.tensor(block_pcs).to(device)
    return type, block_bbox, block_pcs, block_pcs_torch

def rot_mat(angles):
    return Rotation.from_euler('xyz', angles, degrees=True).as_matrix()

def find_crsp_parts(save_path, gt, gt_pcs, block_type, block_bbox, block_pcs, block_pcs_torch):
    
    pred_pcs, Ts, standard_pcs, ids = [], [], [], []
    
    for _, gt_pc in enumerate(gt_pcs):
        Ts_i = []
        R_gt, t_gt, size_gt = bbox(gt_pc)
        costs = []
        for j, block_pc in enumerate(block_pcs_torch):
            
            R_b, t_b, size_b = block_bbox[j]
            size_diff = np.abs(size_gt.reshape(1, -1) - size_b.reshape(-1, 1))
            _, match_axis = scipy.optimize.linear_sum_assignment(size_diff)
            cost = 0
            for j in range(3):
                cost += size_diff[j, match_axis[j]]
            costs.append(cost)
            R_fix = np.eye(3)
            if (match_axis==np.array([0,2,1])).all():
                R_fix = rot_mat([90,0,0])
            elif (match_axis==np.array([2,1,0])).all():
                R_fix = rot_mat([0,90,0])
            elif (match_axis==np.array([1,0,2])).all():
                R_fix = rot_mat([0,0,90])
            elif (match_axis==np.array([1,2,0])).all():
                R_fix = rot_mat([90,0,0]) @ rot_mat([0,0,90])
            elif (match_axis==np.array([2,0,1])).all():
                R_fix = rot_mat([0,90,0]) @ rot_mat([0,0,90])

            T_can = np.eye(4)
            T_can[:3,:3] = R_b.T
            T_can[:3,3] = - R_b.T @ t_b
            T_fix = np.eye(4)
            T_fix[:3,:3] = R_fix
            T_tar = np.eye(4)
            T_tar[:3,:3] = R_gt
            T_tar[:3,3] = t_gt
    
            T = T_tar @ T_fix @ T_can
            Ts_i.append(T)
        
        min_ind = np.argmin(costs)
        T = Ts_i[min_ind]
        block_pc = block_pcs[min_ind]
        pred_pc = (T[:3,:3] @ block_pc.T + T[:3,3].reshape(-1,1)).T
        quat = Rotation.from_matrix(T[:3,:3]).as_quat()
        
        ids.append(min_ind)
        standard_pcs.append(block_pc)
        Ts.append([*T[:3,3], quat[3], *quat[:3]])
        pred_pcs.append(pred_pc)
    
    pred_pcs = np.array(pred_pcs).reshape(-1, 3)
    targ = torch.tensor(pred_pcs).cuda()
    gt = torch.tensor(gt).cuda()
    dist = (targ.unsqueeze(0) - gt.unsqueeze(1)).norm(dim=2)
    cd = dist.min(dim=0)[0].mean() + dist.min(dim=1)[0].mean()
    if cd > 0.1: return
    ids_set = set(ids)
    ids = np.array(ids)
    eq_class = np.zeros(ids.shape[0])
    for j, id in enumerate(ids_set):
        eq_class[ids == id] = j
    save_path = save_path / block_type
    save_path.mkdir(exist_ok=True)
    
    np.save(save_path / 'parts.npy', np.array(standard_pcs))
    np.save(save_path / 'poses.npy', np.array(Ts))
    np.save(save_path / 'eq_class.npy', eq_class)

if __name__=="__main__":

    root_to_data_folder = wrap_path("dataset/partnet_raw/")
    root_to_meta_folder = "dataset/partnet_dataset/"
    root_to_save_folder = wrap_path("dataset/partnet")
    cats = ['Table', 'Chair', 'Lamp', 'Faucet', 'Display']
    modes = ["test", 'train', 'val']
    levels = [3, 3, 3]
    
    rectangles_data = get_blocks('rectangles')
    spheres_data = get_blocks('spheres')
    non_exact_parts = [rectangles_data, spheres_data]
    
    for cat_name, level in zip(cats, levels):
        # import hier imformation
        fn_hier = root_to_meta_folder + "stats/after_merging_label_ids/" + cat_name + '-hier.txt'
        with open(fn_hier) as f:
            hier = f.readlines()
            hier = {'/'+s.split(' ')[1].replace('\n', ''):int(s.split(' ')[0]) for s in hier}

        # import level information
        fn_level = root_to_meta_folder + "stats/after_merging_label_ids/" + cat_name + '-level-' + str(level) + ".txt"
        lev = [] 
        with open(fn_level) as f:
            lev = f.readlines()
            lev = ['/'+s.split(' ')[1].replace('\n', '') for s in lev]

        # for each mode 
        num = 0
        for mode in modes:
            
            #get the object list to deal with
            object_json =json.load(open(root_to_meta_folder + "stats/train_val_test_split/" + cat_name +"." + mode + ".json"))
            object_list = [int(object_json[i]['anno_id']) for i in range(len(object_json))]
            #for each object:
            for i,fn in enumerate(object_list):

                # get information in obj file
                ret = get_shape_info(root_to_data_folder, fn, lev)
                if ret is None: continue
                parts_pcs, gt_part_pcs, tar_pc, tar_100k, tar_mesh, Rs, ts, parts_names, sizes = ret

                # get class index and geo class index
                for name in parts_names:
                    if name not in hier:
                        hier[name] = len(hier.keys())+1
                parts_ids = [hier[name] for name in parts_names]
                eq_classes = get_eq_classes(sizes, parts_ids)
                
                # get part poses from R , T
                parts_poses = []
                for R, t in zip(Rs, ts):
                    q = Rotation.from_matrix(R).as_quat()
                    quat = np.array([q[3], q[0], q[1], q[2]])
                    parts_pose = np.concatenate((t,quat),axis=0)
                    parts_poses.append(parts_pose)
                parts_poses = np.array(parts_poses)
                
                seg_labels = get_seg_labels(tar_pc, gt_part_pcs)
                tar_100k_label = get_upsampled_pc_label(tar_100k, tar_pc)
                data_path = root_to_save_folder / mode / cat_name / str(fn)
                data_path.mkdir(exist_ok=True, parents=True)
                
                np.save(data_path / 'target.npy', tar_pc)
                np.save(data_path / 'parts.npy', parts_pcs)
                np.save(data_path / 'eq_class.npy', eq_classes)
                np.save(data_path / 'poses.npy', parts_poses)
                np.save(data_path / 'gt_parts.npy', gt_part_pcs)
                np.save(data_path / 'labels.npy', seg_labels)
                np.save(data_path / 'target_100k.npy', tar_100k)
                np.save(data_path / 'target_100k_label.npy', tar_100k_label)
                export_obj(data_path / 'target.obj', *tar_mesh)
                for parts_data in non_exact_parts:
                    find_crsp_parts(data_path, tar_pc, gt_part_pcs, *parts_data)





