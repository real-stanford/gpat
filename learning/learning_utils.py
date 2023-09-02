
import argparse
from utils.utils import *
from utils.chamfer.chamfer import chamfer_distance
import pickle

TAU_P = 0.01
    
def is_success(gt_pcs, pred_pcs, tar_pcs, masks):
        
    B, K, _, _ = gt_pcs.shape
    
    all_part_crt, part_acc, avg_part_cds, total_cds = [], [], [], []
    for b in range(B):
        num_part = K - torch.sum(masks[b]).int()
        pred_pcs_b = pred_pcs[b, :num_part]
        pred_tar_b = pred_pcs_b.reshape(-1, 3)
        part_cds_b = chamfer_distance(gt_pcs[b, :num_part], pred_pcs_b).detach().cpu().numpy()
        avg_part_cds.append(np.mean(part_cds_b))
        all_part_crt.append(int((part_cds_b < TAU_P).all()))
        part_acc.append(np.mean(part_cds_b < TAU_P))
        total_cds.append(chamfer_distance(tar_pcs[b], pred_tar_b)[0].item())
    all_part_crt = np.array(all_part_crt)
    part_acc = np.array(part_acc)
    total_cds = np.array(total_cds)
    avg_part_cds = np.array(avg_part_cds)
    
    metric = {}
    metric[f'cd'] = np.mean(total_cds)
    metric[f'cd_p'] = np.mean(avg_part_cds)
    metric[f'sr'] = np.mean(all_part_crt)
    metric[f'pa'] = np.mean(part_acc)
    
    return metric

def get_cats(mode):

    seen_cats = ['Chair', 'Lamp', 'Faucet']
    unseen_cats = ['Table', 'Display']   

    if mode == 'gen': 
        return unseen_cats
    if mode == 'all':
        cats = seen_cats + unseen_cats
        return cats
    return seen_cats

def init_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cuda', type=int, default=0) # cuda id
    parser.add_argument('--exp', type=str, default=None) # experiment name
    parser.add_argument('--seed', type=int, default=42) # random seed
    
    parser.add_argument('--batch_size', type=int, default=32) # batch_size
    parser.add_argument('--logs_dir', type=str, default='logs') # directory for model checkpoints
    parser.add_argument('--results_dir', type=str, default='results') # directory for model checkpoints
    parser.add_argument('--nepoch', type=int, default=2000) # number of training epohcs
    parser.add_argument('--interval', type=int, default=5) # save interval for training
    
    parser.add_argument('--eval', action='store_true') # bool for evaluation
    parser.add_argument('--neval', type=int, default=-1) # number of samples to evaluate, -1 for all
    parser.add_argument('--nviz', type=int, default=0) # number of samples to visualize
    parser.add_argument('--eval_dir', type=str, default=None) # evaluate specified data sample
    parser.add_argument('--model_path', type=str, default=None) # model path
    
    parser.add_argument('--mode', type=str, default='train') # data mode: 'train', 'test', 'gen', 'all'
    parser.add_argument('--cat', type=str, default='all') # data category: 'all' for all categories
    parser.add_argument('--ratio', type=float, default=0) # data composition, probability of using non-exact parts
    parser.add_argument('--rand', action='store_true') # dataset parameter, whether the pose of the target is random
    
    parser.add_argument('--noaug', action='store_true') # abalation parameter, evaluate the model trained without augmentation
    parser.add_argument('--opt', action='store_true') # optimize using cma with predictions as initial values
    
    args = parser.parse_args()
    
    args.max_num_part = 10 # maximum number of parts the model handle
    args.max_num_part_data = 10 # maximum number of parts in a data sample
    args.min_num_part_data = 2 # minimum number of parts in a data sample
    
    if args.exp is None:
        args.exp = f'{args.mode}-{args.cat}'
        if args.rand: args.exp += '-rand'
        if args.ratio != 0: args.exp += '-st'

    train_root = wrap_path('dataset/partnet/train')
    val_root = wrap_path('dataset/partnet/val')
    test_root = wrap_path('dataset/partnet/test')
    
    if args.cat == 'all': cats = get_cats(args.mode)
    else: cats = [args.cat]
    train_dirs = [f'{train_root}/{cat}' for cat in cats]
    train_dirs += [f'{val_root}/{cat}' for cat in cats]
    test_dirs = [f'{test_root}/{cat}' for cat in cats]
    args.train_data_root = train_dirs
    args.test_data_root = test_dirs
    if args.eval_dir is None:
        if not args.eval:
            args.eval_dir = args.train_data_root
        else:
            args.eval_dir = args.test_data_root
    else:
        args.eval_dir = wrap_path(args.eval_dir)
    
    seed_all(args.seed)
    torch.cuda.set_device(args.cuda)
    
    return args

def init_log_dirs(args, model_name):
    
    args.logs_dir = wrap_path(f'{args.logs_dir}/{model_name}/{args.exp}-{datetime.now().strftime("%m-%d")}')
    args.results_dir = wrap_path(f'{args.results_dir}/{model_name}/{args.exp}')
    
    return args

def get_html_seg(output_dir, count, name):
    cols = ['target', 'input', 'gt', 'pred', 'inc']
    data_html = {}
    for i in range(count):
        data_html[f"{i}_target"] = f"{i}/{i}_target.jpg"
        data_html[f"{i}_input"] = f"{i}/{i}_input.jpg"
        data_html[f"{i}_gt"] = f"{i}/{i}_gt.jpg"
        data_html[f"{i}_pred"] = f"{i}/{i}_pred.jpg"
        data_html[f"{i}_inc"] = f"{i}/{i}_inc.jpg"
    ids = [str(i) for i in range(count)]
    html_visualize(str(output_dir), data_html, ids,
                   cols, title=f'{name}', html_file_name='index.html')

def get_html(output_dir, count, name):
    cols = ['target', 'input', 'gt', 'pred']
    data_html = {}
    for i in range(count):
        data_html[f"{i}_target"] = f"{i}/{i}_target.jpg"
        data_html[f"{i}_input"] = f"{i}/{i}_input.jpg"
        data_html[f"{i}_gt"] = f"{i}/{i}_gt.jpg"
        data_html[f"{i}_pred"] = f"{i}/{i}_pred.jpg"
    ids = [str(i) for i in range(count)]
    html_visualize(str(output_dir), data_html, ids,
                   cols, title=f'{name}', html_file_name='index.html')

def get_html_with_seg(output_dir, count, name):
    cols = ['target', 'input', 'gt', 'pred', 'pred-seg']
    data_html = {}
    for i in range(count):
        data_html[f"{i}_target"] = f"{i}/{i}_target.jpg"
        data_html[f"{i}_input"] = f"{i}/{i}_input.jpg"
        data_html[f"{i}_gt"] = f"{i}/{i}_gt.jpg"
        data_html[f"{i}_pred"] = f"{i}/{i}_pred.jpg"
        data_html[f"{i}_pred-seg"] = f"{i}/{i}_pred-seg.jpg"
    ids = [str(i) for i in range(count)]
    html_visualize(str(output_dir), data_html, ids,
                   cols, title=f'{name}', html_file_name='index.html')

def save_pred(log_dir, input_pcs, tar_pc, gt_pcs, gt_seg, gt_poses, pred_pcs, pred_poses, log):
    
    data = {
        'tar_pc': tar_pc,
        'input_pcs': input_pcs,
        'gt_pcs': gt_pcs,
        'gt_poses': gt_poses,
        'gt_seg': gt_seg,
        'pred_pcs': pred_pcs,
        'pred_poses': pred_poses,
    }
    
    if 'pred_seg' in log.keys():
        data['pred_seg'] = log['pred_seg'].detach().cpu().numpy()
    pickle.dump(data, open(log_dir / 'data.pkl', 'wb'))

def visualize_results(logs_dir, cnt, data, pred_pcs, log, save=True):
    masks = data[3].detach().cpu().numpy()
    num_part = int(len(masks)-np.sum(masks))
    
    input_pcs = data[0].detach().cpu().numpy()[:num_part]
    tar_pc = data[1].detach().cpu().numpy()
    gt_pcs = data[4].detach().cpu().numpy()[:num_part]
    gt_poses = data[5].detach().cpu().numpy()[:num_part]
    gt_seg = data[6].detach().cpu().numpy()
    pred_pcs = pred_pcs.detach().cpu().numpy()[:num_part]
    pred_poses = log['pred_poses'].detach().cpu().numpy()[:num_part]
    
    log_dir = logs_dir / f'{cnt}'
    log_dir.mkdir(exist_ok=True)
        
    _, N, _ = pred_pcs.shape
    colors_parts = np.concatenate([np.zeros(N)+i+1 for i in range(num_part)])
    colors_tar = np.zeros(5000)
    pred_pc = np.concatenate(pred_pcs, axis=0)
    gt_pc = np.concatenate(gt_pcs, axis=0)
    for i in range(num_part):
        input_pcs[i, :, 0] += 0.5 * i
    input_pc = np.concatenate(input_pcs, axis=0)
    
    viz_pointcloud(log_dir / f'{cnt}_target.jpg', tar_pc, scalar2rgb(colors_tar))
    viz_pointcloud(log_dir / f'{cnt}_input.jpg', input_pc, scalar2rgb(colors_parts))
    viz_pointcloud(log_dir / f'{cnt}_gt.jpg', gt_pc, scalar2rgb(colors_parts))
    viz_pointcloud(log_dir / f'{cnt}_pred.jpg', pred_pc, scalar2rgb(colors_parts))
    
    if 'pred_seg' in log.keys():
        pred_seg = log['pred_seg'].detach().cpu().numpy()
        viz_pointcloud(log_dir / f'{cnt}_pred-seg.jpg', tar_pc, scalar2rgb(pred_seg+1))

    if save:
        save_pred(log_dir, input_pcs, tar_pc, gt_pcs, gt_poses, gt_seg, pred_pcs, pred_poses, log)

def annotate(path, rec, miou, acc):
    from PIL import Image, ImageDraw, ImageFont
    myFont = ImageFont.truetype('FreeMono.ttf', 100)
    img = Image.open(path)
    I1 = ImageDraw.Draw(img)
    color = (0, 154, 23)
    I1.text((1300, 30),  f'rec {rec:.3f}', font=myFont, fill=color)
    I1.text((1300, 130),  f'iou {miou:.3f}', font=myFont, fill=color)
    I1.text((1300, 230),  f'acc {acc:.3f}', font=myFont, fill=color)
    I1 = ImageDraw.Draw(img)
    img.save(path)
    
def visualize_results_seg(logs_dir, cnt, data, pred, log, save=True):
    masks = data[3].detach().cpu().numpy()
    num_part = int(len(masks)-np.sum(masks))
    
    us_label = (data[1].unsqueeze(1)-data[1][log['idx']].unsqueeze(0)).norm(dim=2).argmin(dim=1)
    us_label = us_label.detach().cpu().numpy()
    pred_seg_us = pred[us_label]
    new_gt_seg_us = log['new_gt_seg'][us_label]
    
    input_pcs = data[0].detach().cpu().numpy()[:num_part]
    tar_pc = data[1].detach().cpu().numpy()
    gt_seg = data[6].detach().cpu().numpy()
    pred_seg = pred_seg_us.detach().cpu().numpy()
    new_gt_seg = new_gt_seg_us.detach().cpu().numpy()
    inc = np.zeros_like(pred_seg)
    inc[pred_seg!=new_gt_seg] = 1
    
    log_dir = logs_dir / f'{cnt}'
    log_dir.mkdir(exist_ok=True)
        
    _, N, _ = input_pcs.shape
    colors_parts = np.concatenate([np.zeros(N)+i+1 for i in range(num_part)])
    colors_tar = np.zeros(5000)
    for i in range(num_part):
        input_pcs[i, :, 0] += 0.5 * i
    input_pc = np.concatenate(input_pcs, axis=0)
    
    viz_pointcloud(log_dir / f'{cnt}_target.jpg', tar_pc, scalar2rgb(colors_tar))
    viz_pointcloud(log_dir / f'{cnt}_input.jpg', input_pc, scalar2rgb(colors_parts))
    viz_pointcloud(log_dir / f'{cnt}_gt.jpg', tar_pc, scalar2rgb(gt_seg+1))
    viz_pointcloud(log_dir / f'{cnt}_pred.jpg', tar_pc, scalar2rgb(pred_seg+1))
    viz_pointcloud(log_dir / f'{cnt}_inc.jpg', tar_pc, scalar2rgb(inc))
    
    annotate(log_dir / f'{cnt}_pred.jpg', log['recall'], log['miou'], log['acc'])