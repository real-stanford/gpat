
from torch import nn, optim
from learning.gpat.gpat import GPAT
from utils.utils import *
from learning.learning_utils import *
from learning.dataset import AssemblyDataset

class Segmenter():
    def __init__(self, device, logs_dir='logs', args=None):
        self.logs_dir = wrap_path(logs_dir)
        self.device = device
        self.max_num_part = args.max_num_part
        self.model = GPAT().to(device).float()
        self.optim = optim.Adam(self.model.parameters(), lr=4e-5)
    
    def match_gt(self, gt_seg, pred_seg, eq_classes):
        B = gt_seg.shape[0]
        new_gt_seg = gt_seg.clone()
        gt_to_pred_match = torch.arange(self.max_num_part).unsqueeze(0).repeat(B, 1).to(self.device)
        with torch.no_grad():
            for b in range(B):
                classes = eq_classes[b]
                n_part = torch.max(gt_seg[b])+1
                for c in set(classes.detach().cpu().numpy()):
                    c_inds = torch.where(classes==c)[0]
                    num_c = c_inds.shape[0]
                    if num_c < 2 or c==-1: continue
                    cost_matrix = np.zeros((num_c, num_c)) # (r, c): gt_r -> pred_c
                    for i in range(num_c):
                        for j in range(num_c):
                            pseudo_gt_seg = torch.zeros_like(gt_seg[b]) - 1
                            pseudo_gt_seg[gt_seg[b]==c_inds[i]] = c_inds[j]
                            ce = nn.functional.cross_entropy(pred_seg[b], pseudo_gt_seg, ignore_index=-1)
                            cost_matrix[i, j] = ce
                    _, match = scipy.optimize.linear_sum_assignment(cost_matrix)
                    gt_to_pred_match[b, c_inds] = c_inds[match]
                for i in range(n_part):
                    new_gt_seg[b, gt_seg[b]==i] = gt_to_pred_match[b, i]
        return new_gt_seg
            
    
    def calc_loss(self, gt_seg, tar_pc_idx, logits, eq_classes):
        subsampled_gt_seg = torch.empty_like(tar_pc_idx).long()
        B = gt_seg.shape[0]
        for b in range(B):
            subsampled_gt_seg[b] = gt_seg[b, tar_pc_idx[b].long()]
        new_gt_seg = self.match_gt(subsampled_gt_seg, logits, eq_classes)
        pred_seg = logits.argmax(dim=2)
        loss = nn.functional.cross_entropy(logits.permute(0,2,1), new_gt_seg)
        acc = (pred_seg==new_gt_seg).float().mean(dim=1)
        
        mean_recall, miou = [], []
        for b in range(B):
            gt = new_gt_seg[b]
            pred = pred_seg[b]
            part_ids = set(gt.detach().cpu().numpy())
            iou_list, recall_list = [], []
            for id in part_ids:
                itsc = torch.logical_and(gt==id, pred==id).sum()
                union = torch.logical_or(gt==id, pred==id).sum()
                iou_list.append((itsc / (union+1e-6)).item())
            iou_list = np.array(iou_list)
            for p in np.linspace(0.05,0.5,10)-0.05+0.5:
                recall_list.append(np.sum(iou_list >= p)/iou_list.shape[0])
            mean_recall.append(np.mean(recall_list))
            miou.append(np.mean(iou_list))
        
        return loss, acc, mean_recall, miou, pred_seg, new_gt_seg
    
    def forward(self, data_batch, is_train):
        input_pcs, tar_pcs, eq_class, masks, _, _, seg_labels, _, _ = data_batch
        logits, tar_pc_idx = self.model(input_pcs, tar_pcs, masks)
        loss, acc, mean_recall, miou, pred_seg, new_gt_seg = self.calc_loss(seg_labels, tar_pc_idx, logits, eq_class)
        
        metric, info, log = {}, {}, {}
        metric['acc'] = acc.mean().item()
        metric['mr'] = np.mean(mean_recall)
        metric['miou'] = np.mean(miou)
        log['acc'] = acc.detach().cpu().numpy()
        log['recall'] = np.array(mean_recall)
        log['miou'] = np.array(miou)
        log['idx'] = tar_pc_idx.long()
        log['new_gt_seg'] = new_gt_seg
        
        if is_train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        loss = loss.item()
        
        return loss, pred_seg, metric, info, log
    
    def save(self, epoch):
        state = {'model': self.model.state_dict()}
        path = self.logs_dir/f"{epoch}.pth"
        torch.save(state, path)
        return path
    
    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model'])

if __name__ == "__main__":

    model_name = 'GPAT'
    args = init_args()
    args = init_log_dirs(args, model_name)
    
    if args.model_path is None:
        args.model_path = 'logs/pretrained/gpat.pth'
    
    device = torch.device('cuda')
    model = Segmenter(device, logs_dir=args.logs_dir, args=args)
    if args.eval:
        model.load(args.model_path)
        evaluate(device, model, AssemblyDataset, args, visualizer=visualize_results_seg, html_writer=get_html_seg)
    else:
        train(device, model, AssemblyDataset, args)