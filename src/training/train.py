import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
import torchvision # 新增，用于跨尺度 NMS

# 强制关闭 OpenCV 内部多线程与 OpenCL，防止多进程数据加载时 CPU 和内存跑满
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False) 

from rich.console import Console
from rich.prompt import Confirm
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

def plot_history(history, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train']) + 1)
    plt.plot(epochs, history['train'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val'], 'r-', label='Val Pck')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, progress, scaler):
    model.train()
    total_loss = 0.0
    task_id = progress.add_task(f"[cyan]Train Epoch {epoch}", total=len(dataloader))
    
    for batch_idx, (imgs, targets, class_ids) in enumerate(dataloader):
        # imgs 是单张量，targets 和 class_ids 是多尺度列表，需要遍历移动到 device
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        class_ids = [c.to(device) for c in class_ids]
        
        optimizer.zero_grad()
        
        # 开启自动混合精度 (AMP) - 前向传播
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            preds = model(imgs) # preds 也是包含三个尺度输出的列表
            loss, loss_dict = criterion(preds, targets, class_ids)
            
        # 使用 scaler 放大 loss 并进行反向传播
        scaler.scale(loss).backward()
        
        # 在进行梯度裁剪前，必须先 unscale 梯度
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
        
        # 使用 scaler 更新权重
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress.update(task_id, advance=1, description=f"[cyan]Train Epoch {epoch} | Loss: {loss.item():.4f}")
        
    progress.remove_task(task_id)
    return total_loss / len(dataloader)

def calculate_pck(gt_batch, pred_batch, pck_cfg):
    """
    计算批次数据的 PCK (Percentage of Correct Keypoints)
    """
    correct_kpts = 0
    total_kpts = 0
    target_in_range_dist = pck_cfg['target_in_range_dist']
    threshold = pck_cfg['max_pixel_threshold']
    for gt_dets, pred_dets in zip(gt_batch, pred_batch):
        if len(gt_dets) == 0:
            continue
            
        total_kpts += len(gt_dets) * 4 # 每个装甲板4个点
        
        if len(pred_dets) == 0:
            continue
            
        for gt in gt_dets:
            gt_pts = gt[2:].reshape(4, 2)
            gt_center = gt_pts.mean(axis=0)
            
            # 根据中心点距离寻找最佳匹配的预测框
            best_pred = None
            min_dist = float('inf')
            
            for pred in pred_dets:
                pred_pts = pred[2:].reshape(4, 2)
                pred_center = pred_pts.mean(axis=0)
                dist = np.linalg.norm(gt_center - pred_center)
                if dist < min_dist:
                    min_dist = dist
                    best_pred = pred_pts
            
            # 如果找到了匹配的预测框
            if min_dist < target_in_range_dist and best_pred is not None:
                # 计算 4 个角点的独立欧氏距离
                dists = np.linalg.norm(gt_pts - best_pred, axis=1)
                # 统计距离小于阈值的点数
                correct_kpts += np.sum(dists < threshold)
                
    return correct_kpts, total_kpts

def process_multi_scale_dets(preds, targets, class_ids, strides, input_size, reg_max, conf_thresh, nms_thresh):
    """
    处理多尺度的解码与跨尺度 NMS 融合
    """
    batch_size = preds[0].size(0)
    gt_dets_batch = [[] for _ in range(batch_size)]
    pred_dets_batch = [[] for _ in range(batch_size)]
    
    # 1. 逐个尺度解码
    for i, s in enumerate(strides):
        current_grid = (input_size[0] // s, input_size[1] // s)
        
        gt_scale = decode_tensor(targets[i], is_pred=False, class_tensor=class_ids[i], conf_threshold=0.9, grid_size=current_grid, reg_max=reg_max, img_size=input_size)
        pred_scale = decode_tensor(preds[i], is_pred=True, conf_threshold=conf_thresh, nms_iou_threshold=nms_thresh, grid_size=current_grid, reg_max=reg_max, img_size=input_size)
        
        for b in range(batch_size):
            if len(gt_scale[b]) > 0:
                gt_dets_batch[b].append(gt_scale[b])
            if len(pred_scale[b]) > 0:
                pred_dets_batch[b].append(pred_scale[b])
                
    final_gt_dets = []
    final_pred_dets = []
    
    # 2. 拼接结果并执行跨尺度 NMS
    for b in range(batch_size):
        # 合并 GT
        if len(gt_dets_batch[b]) > 0:
            final_gt_dets.append(np.concatenate(gt_dets_batch[b], axis=0))
        else:
            final_gt_dets.append([])
            
        # 合并 Pred 并重新应用 NMS
        if len(pred_dets_batch[b]) > 0:
            merged_preds = np.concatenate(pred_dets_batch[b], axis=0)
            
            # 构建用于 torchvision.ops.nms 的 Tensor
            scores = torch.tensor(merged_preds[:, 0])
            pts = torch.tensor(merged_preds[:, 2:]).view(-1, 4, 2)
            min_xy, _ = torch.min(pts, dim=1)
            max_xy, _ = torch.max(pts, dim=1)
            boxes = torch.cat([min_xy, max_xy], dim=1)
            
            keep = torchvision.ops.nms(boxes, scores, nms_thresh)
            final_pred_dets.append(merged_preds[keep.numpy()])
        else:
            final_pred_dets.append([])
            
    return final_gt_dets, final_pred_dets

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, progress, input_size, strides, reg_max, conf_thresh, nms_thresh, pck_cfg):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_kpts = 0
    
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
        gt_dets, pred_dets = process_multi_scale_dets(preds, targets, class_ids, strides, input_size, reg_max, conf_thresh, nms_thresh)
        
        # 计算 PCK@0.5
        correct, total = calculate_pck(gt_dets, pred_dets, pck_cfg)
        total_correct += correct
        total_kpts += total
        
        progress.update(task_id, advance=1)
        
    progress.remove_task(task_id)
    
    avg_loss = total_loss / len(dataloader)
    pck_accuracy = (total_correct / total_kpts) if total_kpts > 0 else 0.0
    
    return avg_loss, pck_accuracy

@torch.no_grad()
def visualize_predictions(model, dataloader, device, save_dir, prefix, progress, input_size, strides, reg_max, num_samples=5, conf_threshold=0.5, nms_iou_threshold=0.45):
    model.eval()
    count = 0
    task_id = progress.add_task(f"[yellow]导出 {prefix} 图像...", total=num_samples)
    
    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)
        targets = [t.to(device) for t in targets]
        class_ids = [c.to(device) for c in class_ids]
        preds = model(imgs) 
        
        # 多尺度解码与融合
        gt_dets, pred_dets = process_multi_scale_dets(preds, targets, class_ids, strides, input_size, reg_max, conf_threshold, nms_iou_threshold)
        
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
    nms_thresh = float(post_cfg.get('nms_iou_threshold', 0.45))

    cache_load = cache_loader.get('load', False)
    cache_load_device = cache_loader.get('device', 'cpu')

    early_stop_cfg = train_cfg.get('early_stopping', {})
    auto_stop_enabled = early_stop_cfg.get('enabled', False)
    min_pck = float(early_stop_cfg.get('min_pck', 0.98))

    device = torch.device("cuda" if torch.cuda.is_available() and train_cfg['device'] == 'auto' else train_cfg['device'])
    save_dir = Path(train_cfg.get('save_dir', "./model_res"))
    
    if save_dir.exists() and any(save_dir.iterdir()):
        overwrite = Confirm.ask(f"[bold yellow]输出文件夹 '{save_dir}' 已存在且包含文件，是否清空并刷新？[/bold yellow]")
        if overwrite:
            shutil.rmtree(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            console.print("[green]已清空历史文件夹。[/green]")
        else:
            console.print("[bold red]已取消训练任务以保护现有文件。[/bold red]")
            return
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = train_cfg['epochs']
    history = {'train': [], 'val': [], 'lr': []}

    input_size = tuple(train_cfg.get('input_size', [416, 416]))
    # 获取多尺度 strides，默认为 [8, 16, 32]
    strides = train_cfg.get('strides', [8, 16, 32])
    reg_max = int(train_cfg.get('reg_max', 16)) 

    if cache_load:
        cache_dev = torch.device(cache_load_device)
    else:
        cache_dev = None

    scaler = torch.amp.GradScaler(device.type) 
    
    workers = data_cfg['num_workers'] 
    pin_mem = True if device.type == 'cuda' else False
    
    console.print(f"[bold cyan]正在准备数据集 (动态加载模式, workers={workers})...[/bold cyan]")

    train_loader = DataLoader(
        RMArmorDataset(
            data_cfg['train_img_dir'], 
            data_cfg['train_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            strides=strides, # 替换原来的 grid_size
            cache_device=cache_dev,
            force_no_cache=False          
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
            strides=strides, # 替换原来的 grid_size
            cache_device=cache_dev,
            force_no_cache=False
        ),
        batch_size=train_cfg['batch_size'], 
        shuffle=False, 
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=True if workers > 0 else False
    )

    model = RMDetector(reg_max=reg_max).to(device)
    if continue_cfg['enabled']:
        weight_path = Path(continue_cfg['path'])
        if weight_path.exists():
            model.load_state_dict(torch.load(weight_path))
            console.print(f"[bold green]成功加载历史权重：{weight_path}，继续训练。[/bold green]")
        else:
            console.print(f"[bold yellow]警告：未找到指定的权重文件 {weight_path}，将从头开始训练。[/bold yellow]")

    # 注意：新的 Loss 已动态处理 grid_size，这里无需再传入固定 grid_size
    criterion = RMDetLoss(
        loss_cfg['lambda_conf'], 
        loss_cfg['lambda_box'], 
        loss_cfg['lambda_pose'],
        loss_cfg['lambda_cls'],
        loss_cfg['alpha'],
        loss_cfg['gamma'],
        reg_max=reg_max
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

    best_val_pck = 0.0      

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
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, progress, scaler)
            # 传入多尺度 strides
            val_loss, val_pck = validate(model, val_loader, criterion, device, epoch, progress, input_size, strides, reg_max, conf_thresh, nms_thresh, pck_cfg)
            
            if epoch <= warmup_epochs:
                current_lr = base_lr * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

            history['train'].append(train_loss)
            history['val'].append(val_pck)
            history['lr'].append(current_lr)
            
            console.print(f"[bold cyan]Epoch {epoch}/{epochs}[/bold cyan] | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val PCK@0.5: {val_pck:.4f}")

            if val_pck > best_val_pck:
                best_val_pck = val_pck
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                console.print(f"[green]  -> 发现更高 PCK: {val_pck:.4f}，模型已保存。[/green]")
            
            progress.update(epoch_task, advance=1)
            
            if auto_stop_enabled and val_pck >= min_pck:
                console.print(f"\n[bold yellow]验证集 PCK ({val_pck:.4f}) 已达到设定的停止阈值，提前终止训练。[/bold yellow]")
                break

        torch.save(model.state_dict(), save_dir / "last_model.pth")
        plot_history(history, save_dir / "loss_curve.png")
        
        log_file = save_dir / "train_log.txt"
        with log_file.open("w", encoding="utf-8") as f:
            f.write("Epoch\tLR\tTrain_Loss\tVal_PCK\n")
            for i in range(len(history['train'])):
                f.write(f"{i+1}\t{history['lr'][i]:.6f}\t{history['train'][i]:.6f}\t{history['val'][i]:.6f}\n")
                
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
        
        visualize_predictions(model, vis_train_loader, device, save_dir, prefix="train", progress=progress, input_size=input_size, strides=strides, reg_max=reg_max, num_samples=5, conf_threshold=conf_thresh, nms_iou_threshold=nms_thresh)
        visualize_predictions(model, vis_val_loader, device, save_dir, prefix="val", progress=progress, input_size=input_size, strides=strides, reg_max=reg_max, num_samples=5, conf_threshold=conf_thresh, nms_iou_threshold=nms_thresh)

    console.print(f"\n[bold green]训练与评估完成！所有结果已保存至: {save_dir.absolute()}[/bold green]")

if __name__ == "__main__":
    main()