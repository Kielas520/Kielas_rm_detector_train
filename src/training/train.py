import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import cv2
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
from src.datasets import RMArmorDataset
from src.model import RMDetector, decode_tensor
from src.loss import RMDetLoss

console = Console()

def plot_history(history, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train']) + 1)
    plt.plot(epochs, history['train'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, progress):
    model.train()
    total_loss = 0.0
    task_id = progress.add_task(f"[cyan]Train Epoch {epoch}", total=len(dataloader))
    
    for batch_idx, (imgs, targets, class_ids) in enumerate(dataloader):
        imgs, targets, class_ids =  imgs.to(device), targets.to(device), class_ids.to(device)
        
        optimizer.zero_grad()
        preds = model(imgs)
        loss, loss_dict = criterion(preds, targets, class_ids)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
        optimizer.step()
        
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
            # 原本 decode_tensor 返回的是 [置信度, 8个关键点坐标]（共 9 个值），所以你用 [1:] 切片正好能拿出 8 个坐标去 reshape 成 (4, 2)。
            # 现在它返回的是 [置信度, 类别ID, 8个关键点坐标]（共 10 个值），你再用 [1:] 切片就会拿到 9 个值，当然无法 reshape 成 4x2 的矩阵了。
            # ---> 修改这里：从索引 2 开始切片 <---
            gt_pts = gt[2:].reshape(4, 2)
            gt_center = gt_pts.mean(axis=0)
            
            # 根据中心点距离寻找最佳匹配的预测框
            best_pred = None
            min_dist = float('inf')
            
            for pred in pred_dets:
                # ---> 修改这里：从索引 2 开始切片 <---
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

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, progress, input_size, grid_size, conf_thresh, nms_thresh, pck_cfg):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_kpts = 0
    
    task_id = progress.add_task(f"[magenta]Val Epoch {epoch}", total=len(dataloader))
    
    for imgs, targets, class_ids in dataloader:
        imgs, targets, class_ids = imgs.to(device), targets.to(device), class_ids.to(device)
        preds = model(imgs)
        # 修复传参遗漏：以前只有两个参数，现在必须传 class_ids
        loss, _ = criterion(preds, targets, class_ids)
        total_loss += loss.item()
        
        # 解码张量以获取实际像素坐标
        gt_dets = decode_tensor(targets, is_pred=False, conf_threshold=0.9, grid_size=grid_size, img_size=input_size)
        pred_dets = decode_tensor(preds, is_pred=True, conf_threshold=conf_thresh, nms_iou_threshold=nms_thresh, grid_size=grid_size, img_size=input_size)
        
        # 计算 PCK@0.5
        correct, total = calculate_pck(gt_dets, pred_dets, pck_cfg)
        total_correct += correct
        total_kpts += total
        
        progress.update(task_id, advance=1)
        
    progress.remove_task(task_id)
    
    avg_loss = total_loss / len(dataloader)
    # 如果没有真实标签，避免除以 0
    pck_accuracy = (total_correct / total_kpts) if total_kpts > 0 else 0.0
    
    return avg_loss, pck_accuracy

@torch.no_grad()
def visualize_predictions(model, dataloader, device, save_dir, prefix, progress, input_size, grid_size, num_samples=5, conf_threshold=0.5, nms_iou_threshold=0.45):
    model.eval()
    count = 0
    task_id = progress.add_task(f"[yellow]导出 {prefix} 图像...", total=num_samples)
    
    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        class_ids = class_ids.to(device) # 确保 class_ids 在设备上
        preds = model(imgs) 
        
        # 传入 class_ids 用于真实标签的正确解码
        gt_dets = decode_tensor(targets, is_pred=False, class_tensor=class_ids, conf_threshold=0.9, grid_size=grid_size, img_size=input_size)
        pred_dets = decode_tensor(preds, is_pred=True, conf_threshold=conf_threshold, nms_iou_threshold=nms_iou_threshold, grid_size=grid_size, img_size=input_size)
        
        for i in range(imgs.size(0)):
            if count >= num_samples:
                progress.remove_task(task_id)
                return
            
            img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            fig, ax = plt.subplots(1, figsize=(10, 8))
            ax.imshow(img_np)
            
            # ---------------------------
            # 绘制真实标签 (GT - 绿色)
            # ---------------------------
            for det in gt_dets[i]:
                cls_id = int(det[1])
                pts = det[2:].reshape(4, 2)
                cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1]) # 计算装甲板中心点
                
                ax.scatter(pts[:, 0], pts[:, 1], color='lime', s=20, zorder=3)
                ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color='lime', linewidth=2)
                ax.plot([pts[2, 0], pts[3, 0]], [pts[2, 1], pts[3, 1]], color='lime', linewidth=2)
                
                # 真实类别标注 (引向左上方)
                ax.annotate(f"GT ID: {cls_id}",
                            xy=(cx, cy), xycoords='data',
                            xytext=(cx - 60, cy - 40), textcoords='data',
                            color='lime', weight='bold', fontsize=10,
                            arrowprops=dict(arrowstyle="-", color='lime', alpha=0.7),
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="lime", alpha=0.6))
            
            # ---------------------------
            # 绘制模型预测 (Pred - 红色)
            # ---------------------------
            for det in pred_dets[i]:
                score = det[0]
                cls_id = int(det[1])
                pts = det[2:].reshape(4, 2)
                cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1]) # 计算装甲板中心点
                
                ax.scatter(pts[:, 0], pts[:, 1], color='red', s=20, zorder=3)
                ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]], color='red', linewidth=2, linestyle='--')
                ax.plot([pts[2, 0], pts[3, 0]], [pts[2, 1], pts[3, 1]], color='red', linewidth=2, linestyle='--')
                
                # 预测类别标注 (引向右上方)
                ax.annotate(f"Pred ID: {cls_id}",
                            xy=(cx, cy), xycoords='data',
                            xytext=(cx + 60, cy - 40), textcoords='data',
                            color='red', weight='bold', fontsize=10,
                            arrowprops=dict(arrowstyle="-", color='red', alpha=0.7),
                            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="red", alpha=0.6))
                
                # 置信度标注 (引向右下方)
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
    
    # 获取阈值，如果 yaml 里没写则提供默认值
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
    grid_size = tuple(train_cfg.get('grid_size', [13, 13]))

    # ==========================================
    # 数据加载器配置：放弃全量预加载，回归动态加载
    # cache_dev = torch.device('cpu')
    # ==========================================
    if cache_load:
        cache_dev = torch.device(cache_load_device)
    else:
        cache_dev = None 
    
    workers = data_cfg['num_workers'] 
    pin_mem = True if device.type == 'cuda' else False
    
    console.print(f"[bold cyan]正在准备数据集 (动态加载模式, workers={workers})...[/bold cyan]")

    train_loader = DataLoader(
        RMArmorDataset(
            data_cfg['train_img_dir'], 
            data_cfg['train_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            grid_size=grid_size,
            cache_device=cache_dev           
        ),
        batch_size=train_cfg['batch_size'], 
        shuffle=True, 
        num_workers=workers,
        pin_memory=pin_mem,
        # 保持 worker 存活，避免每个 epoch 重新开辟进程带来的 CPU 负担
        persistent_workers=True if workers > 0 else False
    )
    
    val_loader = DataLoader(
        RMArmorDataset(
            data_cfg['val_img_dir'], 
            data_cfg['val_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            grid_size=grid_size,
            cache_device=cache_dev
        ),
        batch_size=train_cfg['batch_size'], 
        shuffle=False, 
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=True if workers > 0 else False
    )

    model = RMDetector().to(device)
    
    criterion = RMDetLoss(
        loss_cfg['lambda_conf'], 
        loss_cfg['lambda_box'], 
        loss_cfg['lambda_pose'],
        loss_cfg['lambda_cls'],
        loss_cfg['pos_weight'],
        grid_size=grid_size
    ).to(device)
    
    # ==========================================
    # 提取学习率相关配置
    # ==========================================
    lr_cfg = train_cfg['learning_rate']
    base_lr = float(lr_cfg['base_lr'])
    lr_patience = int(lr_cfg.get('patience', 3))
    lr_factor = float(lr_cfg.get('factor', 0.5))

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=base_lr, 
        weight_decay=float(train_cfg['weight_decay'])
    )

    warmup_epochs = max(1, int(epochs * 0.05))

    # 定义基于 PCK 的调度器 (监控指标越大越好)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',          # 这里改为 max，因为 PCK 越高越好
        factor=lr_factor,    
        patience=lr_patience,
        min_lr=1e-6          
    )

    best_val_pck = 0.0       # 替换原有的 best_val_loss = float('inf')

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
            # --- 1. 执行训练和验证 ---
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, progress)
            # 传入新增的解码参数
            val_loss, val_pck = validate(model, val_loader, criterion, device, epoch, progress, input_size, grid_size, conf_thresh, nms_thresh, pck_cfg)
            
            # --- 2. 学习率调度逻辑 ---
            if epoch <= warmup_epochs:
                current_lr = base_lr * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                # 改为监控 PCK 指标
                scheduler.step(val_pck)
                current_lr = optimizer.param_groups[0]['lr']

            # --- 3. 记录与打印 ---
            history['train'].append(train_loss)
            history['val'].append(val_pck) # history 这里为了方便直接存 pck
            history['lr'].append(current_lr)
            
            console.print(f"[bold cyan]Epoch {epoch}/{epochs}[/bold cyan] | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val PCK@0.5: {val_pck:.4f}")
            
            # 核心修改：以 PCK 最高作为保存最佳权重的依据
            if val_pck > best_val_pck:
                best_val_pck = val_pck
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                console.print(f"[green]  -> 发现更高 PCK: {val_pck:.4f}，模型已保存。[/green]")
            
            progress.update(epoch_task, advance=1)
            
            # 早停机制也建议根据 PCK 来，比如连续 N 个 epoch PCK 达到 0.98 以上即可停止
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
        # 1. 显式删除带有 persistent_workers 的 DataLoader 并强制垃圾回收，等待后台进程安全退出
        del train_loader
        del val_loader
        gc.collect()

        console.print("\n[bold cyan]正在生成识别效果可视化图片...[/bold cyan]")
        model.load_state_dict(torch.load(save_dir / "best_model.pth"))
        
        # 2. 重新实例化 Dataset，彻底阻断与之前多进程上下文的联系
        vis_train_dataset = RMArmorDataset(
            data_cfg['train_img_dir'], 
            data_cfg['train_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            grid_size=grid_size,
            cache_device=cache_dev
        )
        
        vis_val_dataset = RMArmorDataset(
            data_cfg['val_img_dir'], 
            data_cfg['val_label_dir'],
            data_cfg['class_id'],
            input_size=input_size, 
            grid_size=grid_size,
            cache_device=cache_dev
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
        
        visualize_predictions(model, vis_train_loader, device, save_dir, prefix="train", progress=progress, input_size=input_size, grid_size=grid_size, num_samples=5, conf_threshold=conf_thresh, nms_iou_threshold=nms_thresh)
        visualize_predictions(model, vis_val_loader, device, save_dir, prefix="val", progress=progress, input_size=input_size, grid_size=grid_size, num_samples=5, conf_threshold=conf_thresh, nms_iou_threshold=nms_thresh)

    console.print(f"\n[bold green]训练与评估完成！所有结果已保存至: {save_dir.absolute()}[/bold green]")

if __name__ == "__main__":
    main()