

from utils.utils import *
from learning.learning_utils import *
from learning.dataset import AssemblyDataset
from learning.loss_reg import calc_loss
from learning.opt.optimizer import optimize_pose

class Optimizer():
    def optimize(self, part_pcs, tar_pcs, masks):
        B, K, _, _ = part_pcs.shape
        pred_poses = torch.empty(B, K, 7).to(part_pcs.device)
        for b in range(B):
            num_part = (K-masks[b].sum()).int()
            pred_poses[b, :num_part] = optimize_pose(tar_pcs[b], part_pcs[b, :num_part], num_part)
        return pred_poses
    
    def forward(self, data_batch, is_train):
        part_pcs, tar_pcs, eq_class, masks, gt_pcs, gt_poses, _, _, _  = data_batch
        pred_poses = self.optimize(part_pcs, tar_pcs, masks)
        
        _, _, _, _, _, _, new_gt_pcs, _, pred_pcs = calc_loss(
            part_pcs, tar_pcs, eq_class, masks, gt_pcs, gt_poses, pred_poses)
        metric = is_success(new_gt_pcs, pred_pcs, tar_pcs, masks)
        
        info, log = {}, {}
        pred = apply_pose(part_pcs, pred_poses)
        log['pred_poses'] = pred_poses
        
        return 0, pred, metric, info, log

if __name__ == "__main__":
    
    args = init_args()
    model_name = 'Opt'
    args = init_log_dirs(args, model_name)
    
    device = torch.device('cuda')
    model = Optimizer()
    evaluate(device, model, AssemblyDataset, args, visualizer=visualize_results, html_writer=get_html)
   