
from utils.utils import *
from torch.utils.data._utils.collate import default_collate


def safe_load(path):
    if path.exists():
        return np.load(path)
    return None

class AssemblyDataset():
    def __init__(self, dirs, device, args):
        dirs = np.array(dirs)
        self.device = device
        self.args = args
        self.max_num_part_data = args.max_num_part_data
        self.min_num_part_data = args.min_num_part_data
        self.max_num_part = args.max_num_part
        self.rand = args.rand
        self.ratio = args.ratio
        self.eval = args.eval
        
        n_parts = np.array([np.load(dir_path / 'parts.npy').shape[0] for dir_path in dirs])
        self.dirs = dirs[np.logical_and(n_parts<=self.max_num_part_data, 
                                        n_parts>=self.min_num_part_data)]
        
    def __getitem__(self, idx):
        
        dir_path = self.dirs[idx]
        tar_pc_np = np.load(dir_path / 'target.npy')
        input_pcs_np = np.load(dir_path / 'parts.npy')
        gt_poses_np = safe_load(dir_path / 'poses.npy')
        eq_class_np = safe_load(dir_path / 'eq_class.npy')
        seg_labels_np = safe_load(dir_path / 'labels.npy')
        target_100k_np = safe_load(dir_path / 'target_100k.npy')
        target_100k_label_np = safe_load(dir_path / 'target_100k_label.npy')
        
        if np.random.rand() > 1 - self.ratio:
            if self.eval:
                rand_type = np.load('dataset/rand_type.npy')[idx]
                standard_type = 'rectangles' if  rand_type == 0 else 1
            else:
                standard_type = 'rectangles' if  np.random.rand() > 0.5 else 'spheres'
            if not (dir_path / f'{standard_type}').exists(): 
                if self.eval: return None
            else:
                input_pcs_np = np.load(dir_path / f'{standard_type}/parts.npy')
                gt_poses_np = safe_load(dir_path / f'{standard_type}/poses.npy')
                eq_class_np = safe_load(dir_path / f'{standard_type}/eq_class.npy')
            
        n_part, n_pt, _ = input_pcs_np.shape
        tar_pc = torch.from_numpy(tar_pc_np).to(self.device).float()
        input_pcs = torch.zeros(self.max_num_part, n_pt, 3).to(self.device).float()
        input_pcs[:n_part] = torch.from_numpy(input_pcs_np[:n_part]).to(self.device).float()
        mask = torch.ones(self.max_num_part).to(self.device).float()
        mask[:n_part] = 0
        gt_poses = torch.tensor([[0,0,0,1,0,0,0]]).repeat((self.max_num_part, 1)).to(self.device).float()
        if gt_poses_np is not None:
            gt_poses[:n_part, :] = torch.from_numpy(gt_poses_np[:n_part]).to(self.device).float()
        eq_class = torch.zeros(self.max_num_part).to(self.device).int()-1
        if eq_class_np is not None:
            eq_class[:n_part] = torch.from_numpy(eq_class_np[:n_part]).to(self.device).float()
        seg_labels = torch.zeros(tar_pc.shape[0]).to(self.device).long()
        if seg_labels_np is not None:
            seg_labels = torch.from_numpy(seg_labels_np).to(self.device).long()
        target_100k = torch.rand(100000, 3).to(self.device).float()
        target_100k_label = torch.rand(100000).to(self.device).long()
        if target_100k_np is not None:
            target_100k = torch.from_numpy(target_100k_np).to(self.device).float()
            target_100k_label = torch.from_numpy(target_100k_label_np).to(self.device).long()
       
        if self.rand:
            if self.eval:
                rand_q = torch.from_numpy(np.load('dataset/rand_quats.npy')[idx])
            else:
                rand_q = torch.from_numpy(Rotation.random().as_quat())
            rand_pose = torch.cat([torch.zeros(3), rand_q[3:], rand_q[:3]]).to(self.device).float()
            rand_pose_repeat = rand_pose.unsqueeze(0).repeat(n_part, 1)
            gt_poses[:n_part] = mult_pose(rand_pose_repeat, gt_poses[:n_part])
            tar_pc = apply_pose(tar_pc, rand_pose)
            target_100k = apply_pose(target_100k, rand_pose)
        gt_pcs = input_pcs.clone()
        gt_pcs[:n_part] = apply_pose(input_pcs[:n_part], gt_poses[:n_part])
            
        return input_pcs, tar_pc, eq_class, mask, gt_pcs, gt_poses, \
                seg_labels, target_100k, target_100k_label
    
    def __len__(self):
        return len(self.dirs)
    
    @staticmethod
    def collate(batch):
        batch = [data for data in batch if data is not None]
        if len(batch) == 0:
            return None
        return default_collate(batch)        
    
class DGLDataset(AssemblyDataset):
    def __init__(self, dirs, device, args):
        super().__init__(dirs, device, args)
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        if data is None: return None
        
        input_pcs, tar_pc, eq_class, mask, gt_pcs, gt_poses, _, target_100k, target_100k_label = data
        
        dir_path = self.dirs[idx]
        n_part = np.load(dir_path / 'parts.npy').shape[0]
        
        relation_matrix = torch.zeros(self.max_num_part, self.max_num_part).to(self.device).float()
        relation_matrix[:n_part, :n_part] = 1
        
        instance_label = torch.zeros(self.max_num_part, self.max_num_part).to(self.device).float()
        for c in range(torch.max(eq_class)+1):
            c_ids = torch.where(eq_class==c)[0]
            for i, obj_id in enumerate(c_ids):
                instance_label[obj_id, i] = 1
        
        return input_pcs, tar_pc, eq_class, mask, gt_pcs, gt_poses, relation_matrix, instance_label, \
                mask, target_100k, target_100k_label