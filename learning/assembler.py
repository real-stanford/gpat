
from utils.chamfer.chamfer import chamfer_distance
from utils.utils import *
from learning.learning_utils import *
from learning.gpat.gpat import GPAT
from learning.dataset import AssemblyDataset
from learning.opt.optimizer import optimize_pose

def rot_mat(angles):
    return Rotation.from_euler('xyz', angles, degrees=True).as_matrix()

def bbox_pe(src_pc, tar_pc):
    
    R_tar, t_tar, size_tar = bbox(tar_pc)
    R_src, t_src, size_src = bbox(src_pc)
    
    size_diff = np.abs(size_tar.reshape(1, -1) - size_src.reshape(-1, 1))
    _, match_axis = scipy.optimize.linear_sum_assignment(size_diff)
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
    T_can[:3,:3] = R_src.T
    T_can[:3,3] = - R_src.T @ t_src
    T_fix = np.eye(4)
    T_fix[:3,:3] = R_fix
    T_tar = np.eye(4)
    T_tar[:3,:3] = R_tar
    T_tar[:3,3] = t_tar
    
    Tb = T_fix @ T_can
    pc1 = (Tb[:3,:3] @ src_pc.T + Tb[:3,3].reshape(-1,1)).T
    pc2 = (tar_pc - T_tar[:3,3].reshape(1,-1)) @ T_tar[:3,:3]
    
    T = T_tar @ T_fix @ T_can
    q = tf.matrix_to_quaternion(torch.tensor(T[:3,:3])).float()
    t = torch.tensor(T[:3,3]).float()
    return torch.cat([t, q])
        
class Assembler():
    def __init__(self, device, args):
        self.device = device
        self.max_num_part = args.max_num_part
        self.opt = args.opt
        self.segmenter = GPAT().to(device).float()
            
    def match_gt(self, input_pcs, eq_classes, gt_pcs, gt_poses, gt_seg, pred_poses):
        B, K, N, _ = gt_pcs.shape
        pred_to_gt_match = torch.arange(K).unsqueeze(0).repeat(B, 1).to(self.device)
        new_gt_pcs = torch.zeros_like(gt_pcs)
        new_gt_poses = torch.zeros_like(gt_poses)
        new_gt_seg = torch.zeros_like(gt_seg).long()
        with torch.no_grad():
            pred_pcs = apply_pose(input_pcs, pred_poses)
            for b in range(B):
                classes = eq_classes[b]
                for c in set(classes.detach().cpu().numpy()):
                    num_c = (classes==c).sum().int()
                    if num_c < 2 or c==-1: continue
                    gt = gt_pcs[b, classes==c].unsqueeze(0).repeat(num_c, 1, 1, 1).reshape(num_c*num_c, N, 3)
                    pred = pred_pcs[b, classes==c].unsqueeze(1).repeat(1, num_c, 1, 1).reshape(num_c*num_c, N, 3)
                    cd = chamfer_distance(pred, gt).reshape(num_c, num_c).detach().cpu().numpy()
                    _, gt_inds = scipy.optimize.linear_sum_assignment(cd)
                    pred_to_gt_match[b, classes==c] = torch.where(classes==c)[0][gt_inds]
                new_gt_poses[b] = gt_poses[b, pred_to_gt_match[b]]
                for i in range(K):
                    new_gt_seg[b, gt_seg[b]==pred_to_gt_match[b,i]] = i
                new_gt_pcs[b] = gt_pcs[b, pred_to_gt_match[b]]
        return pred_to_gt_match, new_gt_pcs, new_gt_poses, new_gt_seg
    
    def calc_loss(self, input_pcs, tar_pcs, eq_classes, masks,
                  gt_pcs, gt_poses, gt_seg, pred_poses, pred_seg):
        pred_to_gt_match, new_gt_pcs, new_gt_poses, new_gt_seg = \
            self.match_gt(input_pcs, eq_classes, gt_pcs, gt_poses, gt_seg, pred_poses)
        masks = 1 - masks # flip mask since 1 indicates padding
        seg_acc = torch.sum(new_gt_seg == pred_seg, dim=1) / new_gt_seg.shape[1]
        
        return seg_acc.mean(), new_gt_pcs, new_gt_poses, new_gt_seg
    
    def get_seg_pc(self, seg, part_pcs, target, tar100k, tar100klabel, num_part):
        K, N, _ = part_pcs.shape
        us_seg = True if tar100klabel.sum()!=0 else False
        seg_pcs = torch.empty_like(part_pcs)
        if us_seg:
            seg = seg[tar100klabel]
        for i in range(num_part):
            if us_seg:
                pc_i = tar100k[seg==i, :].clone()
            else:
                pc_i = target[seg==i, :].clone()
            if pc_i.shape[0] > N:
                sample_inds = np.random.choice(pc_i.shape[0], N, replace=False)
                seg_pcs[i] = pc_i[sample_inds].clone()
            elif pc_i.shape[0] == 0:
                seg_pcs[i] = part_pcs[i].clone()
            elif pc_i.shape[0] < N:
                sample_inds = np.random.choice(pc_i.shape[0], N-pc_i.shape[0], replace=True)
                seg_pcs[i] = torch.cat([pc_i, pc_i[sample_inds]], dim=0)
        return seg_pcs

    def pose_estimation(self, src_pcs, tar_pcs, poses, num_part):
        for k in range(num_part):
            tgt = tar_pcs[k].detach().cpu().numpy()
            tgt = prune_pc(tgt)
            src = src_pcs[k].detach().cpu().numpy()
            src = prune_pc(src)
            if tgt.shape[0] == 0:
                pose = torch.tensor([0,0,0,1,0,0,0])
            else:
                pose = bbox_pe(src, tgt)
            poses[k] = pose.to(src_pcs.device)
        return poses
            
    def get_poses(self, tar_pc, crsp_pcs, part_pcs, num_part):
        K, N, _ = part_pcs.shape
        poses = torch.zeros(K, 7).to(tar_pc.device)
        poses = self.pose_estimation(part_pcs, crsp_pcs, poses, num_part)
        if self.opt:
            poses[:num_part] = optimize_pose(
                tar_pc, part_pcs[:num_part], num_part, poses[:num_part]
            )
        return poses
    
    def get_seg(self, tar_pcs, input_pcs, masks):
        B, M, _ = tar_pcs.shape
        pred_logits, tar_pc_idx = self.segmenter(input_pcs, tar_pcs, masks)
        tar_pc_idx = tar_pc_idx.long()
        pred_seg = pred_logits.argmax(dim=2)
        pred_seg_us = torch.empty(B, M).to(pred_seg.device)
        for b in range(B):
            us_label = (
                tar_pcs[b].unsqueeze(1) - tar_pcs[b,tar_pc_idx[b]].unsqueeze(0)
            ).norm(dim=2).argmin(dim=1)
            pred_seg_us[b] = pred_seg[b, us_label].clone()
        return pred_seg, pred_seg_us
    
    def step(self, tar_pc, part_pcs, masks, tar100k, tar100klabel):
        B, K, N, _ = part_pcs.shape
        
        pred_seg, pred_seg_us = self.get_seg(tar_pc, part_pcs, masks)
        seg = pred_seg_us
        
        poses = torch.empty(B, K, 7).to(tar_pc.device)
        crsp_pcs = torch.empty_like(part_pcs)
        for b in range(B):
            num_part = (K-masks[b].sum()).int()
            crsp_pcs[b] = self.get_seg_pc(seg[b], part_pcs[b], tar_pc[b], tar100k[b], tar100klabel[b], num_part)
            poses[b] = self.get_poses(tar_pc[b], crsp_pcs[b], part_pcs[b], num_part)
        
        return poses, seg, pred_seg, crsp_pcs
        
    def forward(self, data_batch, is_train=False):
        input_pcs, tar_pcs, eq_classes, masks, gt_pcs, gt_poses, \
            gt_seg, tar100k, tar100klabel = data_batch
        with torch.no_grad():
            pred_poses, pred_seg, pred_seg_ds, crsp_pcs = \
                self.step(tar_pcs, input_pcs, masks, tar100k, tar100klabel)
            seg_acc, new_gt_pcs, new_gt_poses, new_gt_seg = self.calc_loss(
                input_pcs, tar_pcs, eq_classes, masks,
                gt_pcs, gt_poses, gt_seg, pred_poses, pred_seg)
        
        metric, info, log = {}, {}, {}
        pred_pcs = apply_pose(input_pcs, pred_poses)
        
        metric = is_success(new_gt_pcs, pred_pcs, tar_pcs, masks)
        metric['seg'] = seg_acc.item()
        log['pred_seg'] = pred_seg
        log['pred_poses'] = pred_poses
      
        return 0, pred_pcs, metric, info, log
    
    def load(self, model_path):
        model_state = torch.load(model_path, map_location=self.device)
        self.segmenter.load_state_dict(model_state['model'])

if __name__ == "__main__":
    
    args = init_args()
    
    if args.noaug:
        model_name = 'Ours-noaug'
    else:
        model_name = 'Ours'
    if args.opt: model_name += '+opt'
    args = init_log_dirs(args, model_name)
    
    if args.model_path is None:
        args.model_path = 'logs/pretrained/gpat.pth'
    device = torch.device('cuda')
    model = Assembler(device, args=args)
    model.load(args.model_path)
    evaluate(device, model, AssemblyDataset, args, visualizer=visualize_results, html_writer=get_html_with_seg)