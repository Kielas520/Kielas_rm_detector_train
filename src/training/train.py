import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import torchvision 
import multiprocessing
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time

# =========================================================================
# 【系统级优化】
# 强制关闭 OpenCV 内部多线程与 OpenCL。
# 避免 DataLoader 多进程 (num_workers > 0) 与 OpenCV 冲突导致资源锁死。
# =========================================================================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False) 

from rich.console import Console
from rich.prompt import Confirm, Prompt  
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
import gc

# 导入自定义模块 (确保 AugmentPipeline 能够被正确导入)
from src.training.src import *

console = Console()

def plot_and_save_curve(data, epochs, title, ylabel, save_path, color='b'):
    """通用单图绘制函数"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data, color=color, linestyle='-', label=title, linewidth=2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def save_training_curves(history, save_dir):
    """保存多种指标曲线，方便直观分析模型收敛情况"""
    epochs = range(1, len(history['val_pck']) + 1)
    
    plot_and_save_curve(history['train_total'], epochs, 'Total Training Loss', 'Loss', save_dir / "loss_total.png", 'b')
    plot_and_save_curve(history['train_pose'], epochs, 'Pose (Keypoints) Loss', 'Loss', save_dir / "loss_pose.png", 'purple')
    plot_and_save_curve(history['train_cls'], epochs, 'Classification Loss', 'Loss', save_dir / "loss_cls.png", 'brown')
    plot_and_save_curve(history['val_pck'], epochs, 'Validation PCK@0.5', 'PCK', save_dir / "val_pck.png", 'r')

# ================= 修改 1：函数签名增加 aug_pipeline 参数 =================
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, progress, scaler, current_stage, processed_counter, batch_size, total_samples, aug_pipeline):
    model.train()
    
    epoch_losses = {'loss_pose': 0.0, 'loss_cls': 0.0, 'total_loss': 0.0}
    task_id = progress.add_task(f"[cyan]Train Epoch {epoch} [bold magenta](Stage {current_stage})[/bold magenta]", total=len(dataloader))

    max_cache = (dataloader.num_workers * dataloader.prefetch_factor) if dataloader.num_workers > 0 else 0

    fetch_start_time = time.time()
    for batch_idx, (imgs, targets, class_ids) in enumerate(dataloader):
        data_time = time.time() - fetch_start_time
        
        # 1. 基础数据推入显存
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        class_ids = [c.to(device) for c in class_ids]
        
        # ================= 修改 2：执行 GPU 级像素增强 =================
        # 此时 imgs 已经在 GPU 上，调用管线进行色彩、光晕、模糊等极速张量运算
        if aug_pipeline is not None:
            imgs = aug_pipeline.process_gpu(imgs)
        # ============================================================

        optimizer.zero_grad()
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            preds = model(imgs) 
            loss, loss_dict = criterion(preds, targets, class_ids)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
        scaler.step(optimizer)
        scaler.update()
        
        for k in epoch_losses:
            epoch_losses[k] += loss_dict[k]
            
        # 队列深度计算
        total_consumed = (epoch - 1) * total_samples + (batch_idx + 1) * batch_size
        with processed_counter.get_lock():
            total_produced = processed_counter.value
            
        cached_batches = max(0, (total_produced - total_consumed) // batch_size)
        cache_color = "red" if cached_batches == 0 else "green"
            
        progress.update(
            task_id, 
            advance=1, 
            description=(
                f"[cyan]Epoch {epoch} [Stage {current_stage}][/cyan] | "
                f"[{cache_color}]CPU缓存: {cached_batches}/{max_cache}批[/{cache_color}] | "
                f"等待数据: {data_time:.2f}s | "
                f"Loss: {loss.item():.3f}"
            )
        )
        
        fetch_start_time = time.time()
        
    progress.remove_task(task_id)
    
    num_batches = len(dataloader)
    for k in epoch_losses:
        epoch_losses[k] /= num_batches
        
    return epoch_losses

def calculate_pck(gt_batch, pred_batch, pck_cfg):
    """计算批次数据的 PCK 与 类别 ID 准确率"""
    correct_kpts = 0
    total_kpts = 0
    correct_ids = 0  
    total_gts = 0    
    
    target_in_range_dist = pck_cfg['target_in_range_dist']
    threshold = pck_cfg['max_pixel_threshold']
    
    for gt_dets, pred_dets in zip(gt_batch, pred_batch):
        if len(gt_dets) == 0:
            continue
            
        total_kpts += len(gt_dets) * 4 
        total_gts += len(gt_dets)      
        
        if len(pred_dets) == 0:
            continue
            
        matched_preds = set()
            
        for gt in gt_dets:
            gt_pts = gt[2:].reshape(4, 2)
            gt_center = gt_pts.mean(axis=0)
            
            best_pred_idx = -1
            min_dist = float('inf')
            best_pred_pts = None
            
            for i, pred in enumerate(pred_dets):
                if i in matched_preds:
                    continue 
                    
                pred_pts = pred[2:].reshape(4, 2)
                pred_center = pred_pts.mean(axis=0)
                dist = np.linalg.norm(gt_center - pred_center)
                
                if dist < min_dist:
                    min_dist = dist
                    best_pred_idx = i
                    best_pred_pts = pred_pts
            
            if min_dist < target_in_range_dist and best_pred_idx != -1:
                matched_preds.add(best_pred_idx) 
                
                dists = np.linalg.norm(gt_pts - best_pred_pts, axis=1)
                correct_kpts += np.sum(dists < threshold)
                
                if int(gt[1]) == int(pred_dets[best_pred_idx][1]):
                    correct_ids += 1
                
    return correct_kpts, total_kpts, correct_ids, total_gts

def process_multi_scale_dets(preds, targets, class_ids, strides, input_size, reg_max, conf_thresh, kpt_dist_thresh):
    """处理并合并多尺度预测结果，执行 NMS"""
    batch_size = preds[0].size(0)
    gt_dets_batch = [[] for _ in range(batch_size)]
    pred_dets_batch = [[] for _ in range(batch_size)]
    
    for i, s in enumerate(strides):
        current_grid = (input_size[0] // s, input_size[1] // s)
        
        gt_scale = decode_tensor(targets[i], is_pred=False, class_tensor=class_ids[i], conf_threshold=0.9, kpt_dist_thresh=kpt_dist_thresh, grid_size=current_grid, reg_max=reg_max, img_size=input_size)
        pred_scale = decode_tensor(preds[i], is_pred=True, conf_threshold=conf_thresh, kpt_dist_thresh=kpt_dist_thresh, grid_size=current_grid, reg_max=reg_max, img_size=input_size)
        
        for b in range(batch_size):
            if len(gt_scale[b]) > 0:
                gt_dets_batch[b].append(gt_scale[b])
            if len(pred_scale[b]) > 0:
                pred_dets_batch[b].append(pred_scale[b])
                
    final_gt_dets = []
    final_pred_dets = []
    
    for b in range(batch_size):
        if len(gt_dets_batch[b]) > 0:
            merged_gts = np.concatenate(gt_dets_batch[b], axis=0)
            gt_scores = torch.ones(merged_gts.shape[0])
            gt_pts = torch.tensor(merged_gts[:, 2:]).view(-1, 4, 2)
            gt_min_xy, _ = torch.min(gt_pts, dim=1)
            gt_max_xy, _ = torch.max(gt_pts, dim=1)
            gt_boxes = torch.cat([gt_min_xy, gt_max_xy], dim=1)
            
            keep_gt = torchvision.ops.nms(gt_boxes, gt_scores, 0.1)
            final_gt_dets.append(merged_gts[keep_gt.numpy()])
        else:
            final_gt_dets.append([])
            
        if len(pred_dets_batch[b]) > 0:
            merged_preds = np.concatenate(pred_dets_batch[b], axis=0)
            scores = torch.tensor(merged_preds[:, 0])
            pts = torch.tensor(merged_preds[:, 2:]) 
            
            keep = keypoint_nms(pts, scores, kpt_dist_thresh)
            final_pred_dets.append(merged_preds[keep.numpy()])
        else:
            final_pred_dets.append([])
            
    return final_gt_dets, final_pred_dets

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, progress, input_size, strides, reg_max, conf_thresh, kpt_dist_thresh, pck_cfg):
    """验证集评估，注意验证集绝对不使用任何在线数据增强"""
    model.eval()
    total_loss = 0.0
    total_correct_kpts = 0
    total_kpts = 0
    total_correct_ids = 0 
    total_gts = 0         
    
    task_id = progress.add_task(f"[magenta]Val Epoch {epoch}", total=len(dataloader))

    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        class_ids = [c.to(device) for c in class_ids]
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            preds = model(imgs)
            loss, _ = criterion(preds, targets, class_ids)
            
        total_loss += loss.item()
        
        gt_dets, pred_dets = process_multi_scale_dets(preds, targets, class_ids, strides, input_size, reg_max, conf_thresh, kpt_dist_thresh)
        correct_k, total_k, correct_id, total_gt = calculate_pck(gt_dets, pred_dets, pck_cfg)
        
        total_correct_kpts += correct_k
        total_kpts += total_k
        total_correct_ids += correct_id
        total_gts += total_gt
        
        progress.update(task_id, advance=1)
        
    progress.remove_task(task_id)
    
    avg_loss = total_loss / len(dataloader)
    pck_accuracy = (total_correct_kpts / total_kpts) if total_kpts > 0 else 0.0
    id_accuracy = (total_correct_ids / total_gts) if total_gts > 0 else 0.0
    
    return avg_loss, pck_accuracy, id_accuracy

@torch.no_grad()
def visualize_predictions(model, dataloader, device, save_dir, prefix, progress, input_size, strides, reg_max, num_samples=5, conf_threshold=0.5, kpt_dist_thresh=14.5, aug_pipeline=None):
    """输出可视化效果图，便于人工检验模型识别精度"""
    model.eval()
    count = 0
    task_id = progress.add_task(f"[yellow]导出 {prefix} 图像...", total=num_samples)
    
    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)

        # 2. 补充 GPU 增强调用 (仅在查看 train 样本且存在管线时触发)
        if prefix == "train" and aug_pipeline is not None:
            imgs = aug_pipeline.process_gpu(imgs)

        targets = [t.to(device) for t in targets]
        class_ids = [c.to(device) for c in class_ids]
        preds = model(imgs) 
        
        gt_dets, pred_dets = process_multi_scale_dets(preds, targets, class_ids, strides, input_size, reg_max, conf_threshold, kpt_dist_thresh)
        
        for i in range(imgs.size(0)):
            if count >= num_samples:
                progress.remove_task(task_id)
                return
            
            img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(img_np)
            
            if len(gt_dets[i]) > 0:
                for det in gt_dets[i]:
                    cls_id = int(det[1])
                    pts = det[2:].reshape(4, 2)
                    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1]) 
                    ax.scatter(pts[:, 0], pts[:, 1], color='lime', s=20, zorder=3)
                    ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color='lime', linewidth=2)
                    ax.plot([pts[2, 0], pts[3, 0]], [pts[2, 1], pts[3, 1]], color='lime', linewidth=2)
                    ax.annotate(f"GT ID: {cls_id}", xy=(cx, cy), xytext=(cx - 60, cy - 40), 
                                color='lime', weight='bold', fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="lime", alpha=0.6))
            
            if len(pred_dets[i]) > 0:
                for det in pred_dets[i]:
                    score = det[0]
                    cls_id = int(det[1])
                    pts = det[2:].reshape(4, 2)
                    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1]) 
                    ax.scatter(pts[:, 0], pts[:, 1], color='red', s=20, zorder=3)
                    ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color='red', linewidth=2, linestyle='--')
                    ax.plot([pts[2, 0], pts[3, 0]], [pts[2, 1], pts[3, 1]], color='red', linewidth=2, linestyle='--')
                    ax.annotate(f"Pred ID: {cls_id}", xy=(cx, cy), xytext=(cx + 60, cy - 40), 
                                color='red', weight='bold', fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="red", alpha=0.6))
                    ax.annotate(f"Conf: {score:.2f}", xy=(cx, cy), xytext=(cx + 60, cy + 40), 
                                color='red', weight='bold', fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="red", alpha=0.6))
            
            plt.title(f"{prefix} Set - Sample {count+1}\nGreen: GT | Red: Pred")
            plt.axis('off')
            
            save_path = save_dir / f"{prefix}_sample_{count+1}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            count += 1
            progress.update(task_id, advance=1)

# =========================================================================
# 【主函数与训练大循环】
# =========================================================================
def main():
    config_file = Path("./config.yaml")
    if not config_file.exists():
        console.print(f"[bold red]错误：找不到配置文件 {config_file}[/bold red]")
        return

    # 加载配置
    with open(config_file, 'r', encoding='utf-8') as f:
        cfg_data = yaml.safe_load(f)
        cfg = cfg_data['kielas_rm_train']
    
    train_cfg = cfg['train']
    loss_cfg = train_cfg['loss']
    pck_cfg = train_cfg['pck']
    data_cfg = train_cfg['data']
    post_cfg = train_cfg['post_process']
    continue_cfg = train_cfg['continue']
    go_on = False
    
    shuffle_interval = train_cfg.get('shuffle_interval', 5)
    conf_thresh = float(post_cfg.get('conf_threshold', 0.5))
    kpt_dist_thresh = float(post_cfg.get('kpt_dist_thresh', 15.0))
    metric_weights = train_cfg.get('metric_weights', {'pck': 0.6, 'id_acc': 0.4})
    w_pck = float(metric_weights.get('pck', 0.6))
    w_id = float(metric_weights.get('id_acc', 0.4))
    early_stop_cfg = train_cfg.get('early_stopping', {})
    auto_stop_enabled = early_stop_cfg.get('enabled', False)
    min_score = float(early_stop_cfg.get('min_score', 0.98))

    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg['device'] == 'auto' else train_cfg['device'])
    save_dir = Path(train_cfg.get('save_dir', "./model_res"))
    scale_ranges = data_cfg.get('scale_ranges', [[0, 64], [32, 128], [96, 9999]])
    
    # 权重保存目录安全检查
    if save_dir.exists() and any(save_dir.iterdir()):
        choice = Prompt.ask(
            f"\n[bold yellow]输出文件夹 '{save_dir}' 已存在且非空，请选择操作：[/bold yellow]\n"
            "  [1] 继续训练 (保留文件，从历史权重恢复)\n"
            "  [2] 清空并刷新 (删除所有文件，重新开始)\n"
            "  [3] 退出任务",
            choices=["1", "2", "3"],
            default="3"
        )
        
        if choice == "2":
            shutil.rmtree(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            console.print("[green]已清空历史文件夹，全新开始训练。[/green]")
            go_on = False  
        elif choice == "3":
            console.print("[bold red]已取消训练任务。[/bold red]")
            return
        else:
            console.print("[green]选择了继续训练，保留现有文件夹。[/green]")
            go_on = True
            if not continue_cfg.get('path'):
                continue_cfg['path'] = str(save_dir / "last_model.pth")
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = train_cfg['epochs']
    history = {'train_total': [], 'train_pose': [], 'train_cls': [], 'val_pck': [], 'val_id_acc': [], 'val_score': [], 'lr': []}
    input_size = tuple(train_cfg.get('input_size', [416, 416]))
    strides = train_cfg.get('strides', [8, 16, 32])
    reg_max = int(train_cfg.get('reg_max', 16)) 
    scaler = torch.amp.GradScaler(device.type)
    
    workers = data_cfg['num_workers'] 
    pin_mem = True if device.type == 'cuda' else False

    # ---------------- 增强环境初始化 ----------------
    console.print(f"[bold cyan]正在初始化混合数据增强环境(CPU逻辑 + GPU渲染)...[/bold cyan]")
    
    # ================= 修改 3：实例化双向数据增强管线 =================
    aug_cfg_dict = cfg['dataset']['augment']
    aug_cfg = AugmentConfig(**aug_cfg_dict)
    aug_pipeline = AugmentPipeline(aug_cfg)
    # ===============================================================

    bg_dir_str = str(aug_cfg_dict.get('bg_dir', './background'))
    shared_stage = multiprocessing.Value('i', 0)
    processed_counter = multiprocessing.Value('i', 0) 
    
    console.print(f"[bold cyan]准备数据集 (多进程 Worker={workers})...[/bold cyan]")

    # ================= 修改 4：将管线传给 Dataset =================
    train_loader = DataLoader(
        RMArmorDataset(
            data_cfg['train_img_dir'], 
            data_cfg['train_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            strides=strides,
            scale_ranges=scale_ranges, 
            data_name='train',
            aug_pipeline=aug_pipeline,          # 注入管线 (替代原来的 augment_cfg)
            bg_dir=bg_dir_str,           
            shared_stage=shared_stage,
            processed_counter=processed_counter 
        ),
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=True if workers > 0 else False,
        prefetch_factor=train_cfg.get('prefetch_factor', 4) if workers > 0 else None
    )
    
    val_loader = DataLoader(
        RMArmorDataset(
            data_cfg['val_img_dir'], 
            data_cfg['val_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            strides=strides,
            scale_ranges=scale_ranges, 
            data_name='val'
            # 验证集不使用任何增强，保持默认
        ),
        batch_size=train_cfg['batch_size'], 
        shuffle=False, 
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=True if workers > 0 else False
    )

    model = RMDetector(reg_max=reg_max).to(device)
    
    if go_on:
        weight_path = Path(continue_cfg['path'])
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path))
            console.print(f"[bold green]成功加载历史权重：{weight_path}，继续训练。[/bold green]")
        else:
            console.print(f"[bold yellow]警告：指定的历史权重文件 {weight_path} 不存在，将自动从头开始训练。[/bold yellow]")
            go_on = False

    train_img_path = Path(data_cfg['train_img_dir'])
    dataset_yaml_path = train_img_path.parent.parent / "dataset.yaml"
    class_weights_tensor = None
    
    if dataset_yaml_path.exists():
        with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
            ds_cfg = yaml.safe_load(f)
            weights_dict = ds_cfg.get('weights', {})
            if weights_dict:
                num_classes = 12 
                weights_list = [1.0] * num_classes
                for cls_idx, weight in weights_dict.items():
                    if int(cls_idx) < num_classes:
                        weights_list[int(cls_idx)] = float(weight)
                
                class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)
                console.print(f"[bold green]已从 {dataset_yaml_path.name} 加载多类别权重: {weights_list}[/bold green]")
    else:
        console.print(f"[bold yellow]警告：未在 {dataset_yaml_path.parent} 下找到 dataset.yaml，将不使用类别加权。[/bold yellow]")

    criterion = RMDetLoss(
        lambda_pose=loss_cfg.get('lambda_pose', 1.5),
        lambda_cls=loss_cfg.get('lambda_cls', 1.0),
        alpha=loss_cfg.get('alpha', 0.85),
        gamma=loss_cfg.get('gamma', 2.0),
        reg_max=reg_max,
        omega=loss_cfg.get('omega', 10.0),
        epsilon=loss_cfg.get('epsilon', 2.0),
        class_weights=class_weights_tensor
    ).to(device)
    
    optim_cfg = train_cfg['optimizer']
    base_lr = float(optim_cfg['base_lr'])
    betas = optim_cfg['betas']
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=base_lr, 
        betas=betas, 
        weight_decay=float(train_cfg['weight_decay'])
    )

    warmup_epochs = max(1, int(epochs * 0.05))
    scheduler_cfg = train_cfg['scheduler']
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=scheduler_cfg['T_0'], T_mult=scheduler_cfg['T_mult'], eta_min=1e-6)
    best_val_score = 0.0     

    console.print("[bold green]开始训练...[/bold green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"), 
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        epoch_task = progress.add_task(
            f"[bold green]Overall Training | Stage 0 | Next Shuffle: {shuffle_interval} epochs", 
            total=epochs
        )
        
        for epoch in range(1, epochs + 1):
            if epoch > 1 and (epoch - 1) % shuffle_interval == 0:
                with shared_stage.get_lock():
                    shared_stage.value += 1
                console.print(f"\n[bold yellow]🔄 触发半在线洗牌机制！目前进入 Stage {shared_stage.value}[/bold yellow]")
            
            next_shuffle_in = shuffle_interval - ((epoch - 1) % shuffle_interval)
            
            progress.update(
                epoch_task, 
                description=f"[bold green]Overall Training | Stage {shared_stage.value} | Next Shuffle: {next_shuffle_in} epochs"
            )
            
            # ================= 修改 5：将 aug_pipeline 传入训练步骤 =================
            epoch_losses = train_one_epoch(
                model, train_loader, optimizer, criterion, device, 
                epoch, progress, scaler, shared_stage.value,
                processed_counter=processed_counter,            
                batch_size=train_cfg['batch_size'],             
                total_samples=len(train_loader.dataset),
                aug_pipeline=aug_pipeline                       # 传入管线用于 GPU 计算
            )
            
            val_loss, val_pck, val_id_acc = validate(
                model, val_loader, criterion, device, epoch, progress, 
                input_size, strides, reg_max, conf_thresh, kpt_dist_thresh, pck_cfg
            )
            
            if epoch <= warmup_epochs:
                current_lr = base_lr * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

            val_score = (w_pck * val_pck) + (w_id * val_id_acc)

            history['train_total'].append(epoch_losses['total_loss'])
            history['train_pose'].append(epoch_losses['loss_pose'])
            history['train_cls'].append(epoch_losses['loss_cls'])
            history['val_pck'].append(val_pck)
            history['val_id_acc'].append(val_id_acc)
            history['val_score'].append(val_score)
            history['lr'].append(current_lr)
            
            console.print(
                f"[bold cyan]Epoch {epoch}/{epochs}[/bold cyan] | "
                f"Train Total: {epoch_losses['total_loss']:.4f} | "
                f"Pose: {epoch_losses['loss_pose']:.4f} | "
                f"Cls: {epoch_losses['loss_cls']:.4f}"
            )
            console.print(
                f"             [bold magenta]↳[/bold magenta] LR: {current_lr:.6f} | "
                f"Val PCK: {val_pck:.4f} | "
                f"Val ID Acc: {val_id_acc:.4f} | "
                f"Val Score: {val_score:.4f}"
            )
            
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                console.print(f"[green]  -> 发现更高综合得分: {val_score:.4f}，模型已保存。[/green]")
            
            progress.update(epoch_task, advance=1)
            
            if auto_stop_enabled and val_score >= min_score:
                console.print(f"\n[bold yellow]验证集综合得分 ({val_score:.4f}) 已达到设定的停止阈值，提前终止训练。[/bold yellow]")
                break

        torch.save(model.state_dict(), save_dir / "last_model.pth")
        save_training_curves(history, save_dir)
        
        log_file = save_dir / "train_log.txt"
        with log_file.open("w", encoding="utf-8") as f:
            f.write("Epoch\tLR\tTotal_Loss\tPose_Loss\tCls_Loss\tVal_PCK\n")
            for i in range(len(history['val_pck'])):
                f.write(f"{i+1}\t{history['lr'][i]:.6f}\t{history['train_total'][i]:.6f}\t"
                        f"{history['train_pose'][i]:.6f}\t{history['train_cls'][i]:.6f}\t{history['val_pck'][i]:.6f}\n")
                
        del train_loader
        del val_loader
        gc.collect()

        console.print("\n[bold cyan]正在生成识别效果可视化图片...[/bold cyan]")

        best_model_path = save_dir / "best_model.pth"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path))
        else:
            console.print("[yellow]警告：未发现最佳模型文件，将使用最后一次迭代的权重进行可视化。[/yellow]")
            model.load_state_dict(torch.load(save_dir / "last_model.pth", weights_only=True))
        
        vis_train_dataset = RMArmorDataset(
            data_cfg['train_img_dir'], data_cfg['train_label_dir'], data_cfg['class_id'],
            input_size=input_size, strides=strides, data_name='vis_train',
            aug_pipeline=aug_pipeline,   # 接入增强管线
            bg_dir=bg_dir_str,           # 接入背景图片目录
            shared_stage=shared_stage    # 同步洗牌种子
        )
        
        vis_val_dataset = RMArmorDataset(
            data_cfg['val_img_dir'], data_cfg['val_label_dir'], data_cfg['class_id'],
            input_size=input_size, strides=strides, data_name='vis_val'
        )

        vis_train_loader = DataLoader(vis_train_dataset, batch_size=1, shuffle=True, num_workers=0)
        vis_val_loader = DataLoader(vis_val_dataset, batch_size=1, shuffle=True, num_workers=0)
        
        try:
            visualize_predictions_with_features(
                model, vis_train_loader, device, save_dir, prefix="train", 
                progress=progress, input_size=input_size, strides=strides, reg_max=reg_max, 
                process_multi_scale_dets_fn=process_multi_scale_dets,
                num_samples=5, conf_threshold=conf_thresh, kpt_dist_thresh=kpt_dist_thresh
            )
            visualize_predictions_with_features(
                model, vis_val_loader, device, save_dir, prefix="val", 
                progress=progress, input_size=input_size, strides=strides, reg_max=reg_max, 
                process_multi_scale_dets_fn=process_multi_scale_dets,
                num_samples=5, conf_threshold=conf_thresh, kpt_dist_thresh=kpt_dist_thresh
            )
        except NameError:
            visualize_predictions(
                model, vis_train_loader, device, save_dir, prefix="train", 
                progress=progress, input_size=input_size, strides=strides, reg_max=reg_max,
                num_samples=5, conf_threshold=conf_thresh, kpt_dist_thresh=kpt_dist_thresh
            )
            visualize_predictions(
                model, vis_val_loader, device, save_dir, prefix="val", 
                progress=progress, input_size=input_size, strides=strides, reg_max=reg_max,
                num_samples=5, conf_threshold=conf_thresh, kpt_dist_thresh=kpt_dist_thresh
            )

    console.print(f"\n[bold green]训练与评估完成！所有结果已保存至: {save_dir.absolute()}[/bold green]")

if __name__ == "__main__":
    main()