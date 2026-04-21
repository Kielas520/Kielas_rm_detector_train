import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import torchvision 

# 强制关闭 OpenCV 内部多线程与 OpenCL，防止多进程数据加载时 CPU 和内存跑满
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False) 

from rich.console import Console
from rich.prompt import Confirm, Prompt  # 新增导入 Prompt 用于多选
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前调用
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
import gc

# 导入模块
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
    """像 YOLO 一样分别保存多种指标曲线"""
    epochs = range(1, len(history['val_pck']) + 1)
    
    # 单独保存各项 Loss 和 指标
    plot_and_save_curve(history['train_total'], epochs, 'Total Training Loss', 'Loss', save_dir / "loss_total.png", 'b')
    plot_and_save_curve(history['train_conf'], epochs, 'Confidence Loss', 'Loss', save_dir / "loss_conf.png", 'orange')
    plot_and_save_curve(history['train_box'], epochs, 'Box Loss', 'Loss', save_dir / "loss_box.png", 'green')
    plot_and_save_curve(history['train_pose'], epochs, 'Pose (Keypoints) Loss', 'Loss', save_dir / "loss_pose.png", 'purple')
    plot_and_save_curve(history['train_cls'], epochs, 'Classification Loss', 'Loss', save_dir / "loss_cls.png", 'brown')
    
    # 验证集评估指标 PCK 单独保存
    plot_and_save_curve(history['val_pck'], epochs, 'Validation PCK@0.5', 'PCK', save_dir / "val_pck.png", 'r')


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, progress, scaler):
    model.train()
    
    # 初始化记录各项 Loss 的字典
    epoch_losses = {
        'loss_conf': 0.0,
        'loss_box': 0.0,
        'loss_pose': 0.0,
        'loss_cls': 0.0,
        'total_loss': 0.0
    }
    
    task_id = progress.add_task(f"[cyan]Train Epoch {epoch}", total=len(dataloader))
    
    for batch_idx, (imgs, targets, class_ids) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        class_ids = [c.to(device) for c in class_ids]
        
        optimizer.zero_grad()
        
        # 开启自动混合精度 (AMP) - 前向传播
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            preds = model(imgs) 
            loss, loss_dict = criterion(preds, targets, class_ids)
            
        # 使用 scaler 放大 loss 并进行反向传播
        scaler.scale(loss).backward()
        
        # 在进行梯度裁剪前，必须先 unscale 梯度
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
        
        # 使用 scaler 更新权重
        scaler.step(optimizer)
        scaler.update()
        
        # 累加各项 Loss
        for k in epoch_losses:
            epoch_losses[k] += loss_dict[k]
            
        progress.update(task_id, advance=1, description=f"[cyan]Train Epoch {epoch} | Total Loss: {loss.item():.4f} | Conf: {loss_dict['loss_conf']:.4f}")
        
    progress.remove_task(task_id)
    
    # 取该 Epoch 各项 Loss 的平均值
    num_batches = len(dataloader)
    for k in epoch_losses:
        epoch_losses[k] /= num_batches
        
    return epoch_losses

def calculate_pck(gt_batch, pred_batch, pck_cfg):
    """
    计算批次数据的 PCK 与 类别 ID 准确率
    """
    correct_kpts = 0
    total_kpts = 0
    correct_ids = 0  # 新增：预测正确的类别数量
    total_gts = 0    # 新增：GT目标总数
    
    target_in_range_dist = pck_cfg['target_in_range_dist']
    threshold = pck_cfg['max_pixel_threshold']
    
    for gt_dets, pred_dets in zip(gt_batch, pred_batch):
        if len(gt_dets) == 0:
            continue
            
        total_kpts += len(gt_dets) * 4 # 每个装甲板4个点
        total_gts += len(gt_dets)      # 统计目标个数
        
        if len(pred_dets) == 0:
            continue
            
        # --- 修复：记录已被匹配的预测框索引 ---
        matched_preds = set()
            
        for gt in gt_dets:
            gt_pts = gt[2:].reshape(4, 2)
            gt_center = gt_pts.mean(axis=0)
            
            # 根据中心点距离寻找最佳匹配的预测框
            best_pred_idx = -1
            min_dist = float('inf')
            best_pred_pts = None
            
            for i, pred in enumerate(pred_dets):
                if i in matched_preds:
                    continue # 跳过已经被其他 GT 匹配掉的预测框
                    
                pred_pts = pred[2:].reshape(4, 2)
                pred_center = pred_pts.mean(axis=0)
                dist = np.linalg.norm(gt_center - pred_center)
                
                if dist < min_dist:
                    min_dist = dist
                    best_pred_idx = i
                    best_pred_pts = pred_pts
            
            # 如果找到了符合条件的预测框
            if min_dist < target_in_range_dist and best_pred_idx != -1:
                matched_preds.add(best_pred_idx) # 标记为已占用
                
                # 计算 4 个角点的独立欧氏距离
                dists = np.linalg.norm(gt_pts - best_pred_pts, axis=1)
                # 统计距离小于阈值的点数
                correct_kpts += np.sum(dists < threshold)
                
                # --- 新增：如果 ID 匹配正确，累加 correct_ids ---
                # gt[1] 和 pred_dets[best_pred_idx][1] 分别是真实类别和预测类别
                if int(gt[1]) == int(pred_dets[best_pred_idx][1]):
                    correct_ids += 1
                
    return correct_kpts, total_kpts, correct_ids, total_gts

def process_multi_scale_dets(preds, targets, class_ids, strides, input_size, reg_max, conf_thresh, kpt_dist_thresh):
    """
    处理多尺度的解码与跨尺度 NMS 融合
    """
    batch_size = preds[0].size(0)
    gt_dets_batch = [[] for _ in range(batch_size)]
    pred_dets_batch = [[] for _ in range(batch_size)]
    
    # 1. 逐个尺度解码
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
    
    # 2. 拼接结果并执行跨尺度 NMS
    for b in range(batch_size):
        # --- 修复：合并 GT 并执行去重 ---
        if len(gt_dets_batch[b]) > 0:
            merged_gts = np.concatenate(gt_dets_batch[b], axis=0)
            
            # 构建用于 NMS 的 Tensor (赋予相同置信度)
            gt_scores = torch.ones(merged_gts.shape[0])
            gt_pts = torch.tensor(merged_gts[:, 2:]).view(-1, 4, 2)
            gt_min_xy, _ = torch.min(gt_pts, dim=1)
            gt_max_xy, _ = torch.max(gt_pts, dim=1)
            gt_boxes = torch.cat([gt_min_xy, gt_max_xy], dim=1)
            
            # 对 GT 使用较严格的 NMS 去重
            keep_gt = torchvision.ops.nms(gt_boxes, gt_scores, 0.1)
            final_gt_dets.append(merged_gts[keep_gt.numpy()])
        else:
            final_gt_dets.append([])
            
        # --- 合并 Pred 并重新应用 Keypoint NMS ---
        if len(pred_dets_batch[b]) > 0:
            merged_preds = np.concatenate(pred_dets_batch[b], axis=0)
            
            scores = torch.tensor(merged_preds[:, 0])
            pts = torch.tensor(merged_preds[:, 2:]) # 形状 [N, 8]
            
            # 使用关键点 NMS，跨尺度重叠也会因为角点距离过近被抑制
            keep = keypoint_nms(pts, scores, kpt_dist_thresh)
            final_pred_dets.append(merged_preds[keep.numpy()])
        else:
            final_pred_dets.append([])
            
    return final_gt_dets, final_pred_dets

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, progress, input_size, strides, reg_max, conf_thresh, kpt_dist_thresh, pck_cfg):
    model.eval()
    total_loss = 0.0
    total_correct_kpts = 0
    total_kpts = 0
    total_correct_ids = 0 # 新增
    total_gts = 0         # 新增
    
    task_id = progress.add_task(f"[magenta]Val Epoch {epoch}", total=len(dataloader))
    
    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        class_ids = [c.to(device) for c in class_ids]
        
        # 验证阶段使用混合精度加速
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            preds = model(imgs)
            loss, _ = criterion(preds, targets, class_ids)
            
        total_loss += loss.item()
        
        # 多尺度解码与融合
        gt_dets, pred_dets = process_multi_scale_dets(preds, targets, class_ids, strides, input_size, reg_max, conf_thresh, kpt_dist_thresh)
        
        # 计算 PCK@0.5 与 ID_Acc
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
def visualize_predictions(model, dataloader, device, save_dir, prefix, progress, input_size, strides, reg_max, num_samples=5, conf_threshold=0.5, kpt_dist_thresh=14.5):
    model.eval()
    count = 0
    task_id = progress.add_task(f"[yellow]导出 {prefix} 图像...", total=num_samples)
    
    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        class_ids = [c.to(device) for c in class_ids]
        preds = model(imgs) 
        
        # 多尺度解码与融合
        gt_dets, pred_dets = process_multi_scale_dets(preds, targets, class_ids, strides, input_size, reg_max, conf_threshold, kpt_dist_thresh)
        
        for i in range(imgs.size(0)):
            if count >= num_samples:
                progress.remove_task(task_id)
                return
            
            img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(img_np)
            
            # 绘制真实标签 (GT - 绿色)
            if len(gt_dets[i]) > 0:
                for det in gt_dets[i]:
                    cls_id = int(det[1])
                    pts = det[2:].reshape(4, 2)
                    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1]) 
                    
                    ax.scatter(pts[:, 0], pts[:, 1], color='lime', s=20, zorder=3)
                    ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color='lime', linewidth=2)
                    ax.plot([pts[2, 0], pts[3, 0]], [pts[2, 1], pts[3, 1]], color='lime', linewidth=2)
                    
                    ax.annotate(f"GT ID: {cls_id}",
                                xy=(cx, cy), xycoords='data',
                                xytext=(cx - 60, cy - 40), textcoords='data',
                                color='lime', weight='bold', fontsize=10,
                                arrowprops=dict(arrowstyle="-", color='lime', alpha=0.7),
                                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="lime", alpha=0.6))
            
            # 绘制模型预测 (Pred - 红色)
            if len(pred_dets[i]) > 0:
                for det in pred_dets[i]:
                    score = det[0]
                    cls_id = int(det[1])
                    pts = det[2:].reshape(4, 2)
                    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1]) 
                    
                    ax.scatter(pts[:, 0], pts[:, 1], color='red', s=20, zorder=3)
                    ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color='red', linewidth=2, linestyle='--')
                    ax.plot([pts[2, 0], pts[3, 0]], [pts[2, 1], pts[3, 1]], color='red', linewidth=2, linestyle='--')
                    
                    ax.annotate(f"Pred ID: {cls_id}",
                                xy=(cx, cy), xycoords='data',
                                xytext=(cx + 60, cy - 40), textcoords='data',
                                color='red', weight='bold', fontsize=10,
                                arrowprops=dict(arrowstyle="-", color='red', alpha=0.7),
                                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="red", alpha=0.6))
                    
                    ax.annotate(f"Conf: {score:.2f}",
                                xy=(cx, cy), xycoords='data',
                                xytext=(cx + 60, cy + 40), textcoords='data',
                                color='red', weight='bold', fontsize=10,
                                arrowprops=dict(arrowstyle="-", color='red', alpha=0.7),
                                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="red", alpha=0.6))
            
            plt.title(f"{prefix} Set - Sample {count+1}\nGreen: GT | Red: Pred")
            plt.axis('off')
            
            save_path = save_dir / f"{prefix}_sample_{count+1}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            count += 1
            progress.update(task_id, advance=1)

def main():
    config_file = Path("./config.yaml")
    if not config_file.exists():
        console.print(f"[bold red]错误：找不到配置文件 {config_file}[/bold red]")
        return

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['kielas_rm_train']
    
    train_cfg = cfg['train']
    loss_cfg = cfg['train']['loss']
    pck_cfg = cfg['train']['pck']
    data_cfg = cfg['train']['data']
    post_cfg = cfg['train']['post_process']
    cache_loader = cfg['train']['cache_loader']
    continue_cfg = cfg['train']['continue']
    
    conf_thresh = float(post_cfg.get('conf_threshold', 0.5))
    kpt_dist_thresh = float(post_cfg.get('kpt_dist_thresh', 15.0))

    cache_load = cache_loader.get('load', False)
    cache_load_device = cache_loader.get('device', 'cpu')

    # 提取权重和停止阈值配置
    metric_weights = train_cfg.get('metric_weights', {'pck': 0.6, 'id_acc': 0.4})
    w_pck = float(metric_weights.get('pck', 0.6))
    w_id = float(metric_weights.get('id_acc', 0.4))

    early_stop_cfg = train_cfg.get('early_stopping', {})
    auto_stop_enabled = early_stop_cfg.get('enabled', False)
    min_score = float(early_stop_cfg.get('min_score', 0.98))

    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg['device'] == 'auto' else train_cfg['device'])
    save_dir = Path(train_cfg.get('save_dir', "./model_res"))
    # 从配置中读取 scale_ranges
    scale_ranges = data_cfg.get('scale_ranges', [[0, 64], [32, 128], [96, 9999]])
    # ---------------- 目录检查与操作询问逻辑修改开始 ----------------
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
            continue_cfg['enabled'] = False  # 强制不继续
        elif choice == "3":
            console.print("[bold red]已取消训练任务。[/bold red]")
            return
        else:
            console.print("[green]选择了继续训练，保留现有文件夹。[/green]")
            continue_cfg['enabled'] = True
            # 如果配置中未指定权重路径，默认使用目录下的 last_model.pth
            if not continue_cfg.get('path'):
                continue_cfg['path'] = str(save_dir / "last_model.pth")
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
    # ---------------- 目录检查与操作询问逻辑修改结束 ----------------
    
    epochs = train_cfg['epochs']
    
    # 修改历史字典记录项
    history = {
            'train_total': [],
            'train_conf': [],
            'train_box': [],
            'train_pose': [],
            'train_cls': [],
            'val_pck': [],
            'val_id_acc': [],
            'val_score': [],
            'lr': []
        }

    input_size = tuple(train_cfg.get('input_size', [416, 416]))
    # 获取多尺度 strides，默认为 [8, 16, 32]
    strides = train_cfg.get('strides', [8, 16, 32])
    reg_max = int(train_cfg.get('reg_max', 16)) 

    if cache_load:
        cache_dev = torch.device(cache_load_device)
    else:
        cache_dev = None

    scaler = torch.amp.GradScaler(device.type) # type: ignore
    
    workers = data_cfg['num_workers'] 
    pin_mem = True if device.type == 'cuda' else False
    
    console.print(f"[bold cyan]正在准备数据集 (动态加载模式, workers={workers})...[/bold cyan]")

    train_loader = DataLoader(
        RMArmorDataset(
            data_cfg['train_img_dir'], 
            data_cfg['train_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            strides=strides,
            scale_ranges=scale_ranges, # 传入参数 
            cache_device=cache_dev,
            force_no_cache=False,
            data_name = 'train'          
        ),
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=True if workers > 0 else False
    )
    
    val_loader = DataLoader(
        RMArmorDataset(
            data_cfg['val_img_dir'], 
            data_cfg['val_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            strides=strides,
            scale_ranges=scale_ranges, # 传入参数 
            cache_device=cache_dev,
            force_no_cache=False,
            data_name = 'val'
        ),
        batch_size=train_cfg['batch_size'], 
        shuffle=False, 
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=True if workers > 0 else False
    )

    model = RMDetector(reg_max=reg_max).to(device)
    
    # ---------------- 权重加载逻辑修改开始 ----------------
    if continue_cfg['enabled']:
        weight_path = Path(continue_cfg['path'])
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path))
            console.print(f"[bold green]成功加载历史权重：{weight_path}，继续训练。[/bold green]")
        else:
            # 捕获文件为空/不存在的情况，自动回退到重新开始训练
            console.print(f"[bold yellow]警告：指定的历史权重文件 {weight_path} 不存在（可能是新文件夹或已被清除），将自动从头开始训练。[/bold yellow]")
            continue_cfg['enabled'] = False
    # ---------------- 权重加载逻辑修改结束 ----------------

    criterion = RMDetLoss(
        loss_cfg['lambda_conf'], 
        loss_cfg['lambda_box'], 
        loss_cfg['lambda_pose'],
        loss_cfg['lambda_cls'],
        loss_cfg['alpha'],
        loss_cfg['gamma'],
        reg_max,
        loss_cfg['omega'],
        loss_cfg['epsilon']
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

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,          
        eta_min=1e-6     
    )

    best_val_score = 0.0 # 以前是 best_val_pck      

    console.print("[bold green]开始训练...[/bold green]")
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        epoch_task = progress.add_task("[bold green]总体训练进度", total=epochs)
        
        for epoch in range(1, epochs + 1):
            
            # 接收带有分类/分项Loss的字典
            epoch_losses = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, progress, scaler)
            val_loss, val_pck, val_id_acc = validate(model, val_loader, criterion, device, epoch, progress, input_size, strides, reg_max, conf_thresh, kpt_dist_thresh, pck_cfg)
            
            if epoch <= warmup_epochs:
                current_lr = base_lr * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

            # --- 新增：计算综合得分 ---
            val_score = (w_pck * val_pck) + (w_id * val_id_acc)

            # 更新至历史记录字典中
            history['train_total'].append(epoch_losses['total_loss'])
            history['train_conf'].append(epoch_losses['loss_conf'])
            history['train_box'].append(epoch_losses['loss_box'])
            history['train_pose'].append(epoch_losses['loss_pose'])
            history['train_cls'].append(epoch_losses['loss_cls'])
            
            history['val_pck'].append(val_pck)
            history['val_id_acc'].append(val_id_acc)
            history['val_score'].append(val_score)
            history['lr'].append(current_lr)
            
            # 分两行打印：第一行打印所有的 loss，第二行打印验证集指标和学习率
            console.print(
                f"[bold cyan]Epoch {epoch}/{epochs}[/bold cyan] | "
                f"Train Total: {epoch_losses['total_loss']:.4f} | "
                f"Conf: {epoch_losses['loss_conf']:.4f} | "
                f"Box: {epoch_losses['loss_box']:.4f} | "
                f"Pose: {epoch_losses['loss_pose']:.4f} | "
                f"Cls: {epoch_losses['loss_cls']:.4f}"
            )
            console.print(
                f"             [bold magenta]↳[/bold magenta] LR: {current_lr:.6f} | "
                f"Val PCK: {val_pck:.4f} | "
                f"Val ID Acc: {val_id_acc:.4f} | "
                f"Val Score: {val_score:.4f}"
            )
            
            # 依据综合得分决定是否保存最优模型
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                console.print(f"[green]  -> 发现更高综合得分: {val_score:.4f}，模型已保存。[/green]")
            
            progress.update(epoch_task, advance=1)
            
            # 依据综合得分判定是否停止
            if auto_stop_enabled and val_score >= min_score:
                console.print(f"\n[bold yellow]验证集综合得分 ({val_score:.4f}) 已达到设定的停止阈值，提前终止训练。[/bold yellow]")
                break

        torch.save(model.state_dict(), save_dir / "last_model.pth")
        
        # 分离保存多张曲线图片
        save_training_curves(history, save_dir)
        
        # 更新日志写入逻辑，将多项损失分别打表记录
        log_file = save_dir / "train_log.txt"
        with log_file.open("w", encoding="utf-8") as f:
            f.write("Epoch\tLR\tTotal_Loss\tConf_Loss\tBox_Loss\tPose_Loss\tCls_Loss\tVal_PCK\n")
            for i in range(len(history['val_pck'])):
                f.write(f"{i+1}\t{history['lr'][i]:.6f}\t{history['train_total'][i]:.6f}\t{history['train_conf'][i]:.6f}\t"
                        f"{history['train_box'][i]:.6f}\t{history['train_pose'][i]:.6f}\t{history['train_cls'][i]:.6f}\t{history['val_pck'][i]:.6f}\n")
                
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
            data_cfg['train_img_dir'], 
            data_cfg['train_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            strides=strides, 
            cache_device=cache_dev,
            force_no_cache=True 
        )
        
        vis_val_dataset = RMArmorDataset(
            data_cfg['val_img_dir'], 
            data_cfg['val_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            strides=strides, 
            cache_device=cache_dev,
            force_no_cache=True
        )

        vis_train_loader = DataLoader(
            vis_train_dataset, 
            batch_size=1, 
            shuffle=True, 
            num_workers=0
        )
        vis_val_loader = DataLoader(
            vis_val_dataset, 
            batch_size=1, 
            shuffle=True, 
            num_workers=0
        )
        
        visualize_predictions(model, vis_train_loader, device, save_dir, prefix="train", progress=progress, input_size=input_size, strides=strides, reg_max=reg_max, num_samples=5, conf_threshold=conf_thresh, kpt_dist_thresh=kpt_dist_thresh)
        visualize_predictions(model, vis_val_loader, device, save_dir, prefix="val", progress=progress, input_size=input_size, strides=strides, reg_max=reg_max, num_samples=5, conf_threshold=conf_thresh, kpt_dist_thresh=kpt_dist_thresh)

    console.print(f"\n[bold green]训练与评估完成！所有结果已保存至: {save_dir.absolute()}[/bold green]")

if __name__ == "__main__":
    main()