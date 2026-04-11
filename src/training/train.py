import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import cv2
# 强制关闭 OpenCV 内部多线程与 OpenCL，防止多进程数据加载时 CPU 和内存跑满
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False) 

from rich.console import Console
from rich.prompt import Confirm
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import shutil

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
        imgs, targets = imgs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        preds = model(imgs)
        loss, loss_dict = criterion(preds, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) 
        optimizer.step()
        
        total_loss += loss.item()
        progress.update(task_id, advance=1, description=f"[cyan]Train Epoch {epoch} | Loss: {loss.item():.4f}")
        
    progress.remove_task(task_id)
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, progress):
    model.eval()
    total_loss = 0.0
    task_id = progress.add_task(f"[magenta]Val Epoch {epoch}", total=len(dataloader))
    
    for imgs, targets, class_ids in dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss, _ = criterion(preds, targets)
        total_loss += loss.item()
        progress.update(task_id, advance=1)
        
    progress.remove_task(task_id)
    return total_loss / len(dataloader)

@torch.no_grad()
def visualize_predictions(model, dataloader, device, save_dir, prefix, progress, input_size, grid_size, num_samples=5):
    model.eval()
    count = 0
    task_id = progress.add_task(f"[yellow]导出 {prefix} 图像...", total=num_samples)
    
    for imgs, targets, class_ids in dataloader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        preds = model(imgs) 
        
        gt_dets = decode_tensor(targets, is_pred=False, conf_threshold=0.9, grid_size=grid_size, img_size=input_size)
        pred_dets = decode_tensor(preds, is_pred=True, conf_threshold=0.5, grid_size=grid_size, img_size=input_size)
        
        for i in range(imgs.size(0)):
            if count >= num_samples:
                progress.remove_task(task_id)
                return
            
            img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            fig, ax = plt.subplots(1, figsize=(8, 6))
            ax.imshow(img_np)
            
            for det in gt_dets[i]:
                pts = det[1:].reshape(4, 2)
                polygon_gt = patches.Polygon(pts, linewidth=2, edgecolor='lime', facecolor='none', closed=True)
                ax.add_patch(polygon_gt)
                
            for det in pred_dets[i]:
                pts = det[1:].reshape(4, 2)
                score = det[0]
                polygon_pred = patches.Polygon(pts, linewidth=2, edgecolor='red', facecolor='none', closed=True, linestyle='--')
                ax.add_patch(polygon_pred)
                ax.text(pts[0,0], pts[0,1]-5, f"{score:.2f}", color='red', fontsize=10, weight='bold')
            
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
    data_cfg = cfg['train']['data']
    
    early_stop_cfg = train_cfg.get('early_stopping', {})
    auto_stop_enabled = early_stop_cfg.get('enabled', False)
    min_val_loss_threshold = float(early_stop_cfg.get('min_val_loss', 0.15))

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
    # ==========================================
    cache_dev = None 
    
    # 之前这里被我写死了 workers = 4，现在改回读取你的配置
    workers = data_cfg['num_workers'] 
    pin_mem = True if device.type == 'cuda' else False
    
    console.print(f"[bold cyan]正在准备数据集 (动态加载模式, workers={workers})...[/bold cyan]")

    train_loader = DataLoader(
        RMArmorDataset(
            data_cfg['train_img_dir'], 
            data_cfg['train_label_dir'],
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
        grid_size=grid_size
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(train_cfg['learning_rate']), 
        weight_decay=float(train_cfg['weight_decay'])
    )
    
    warmup_epochs = max(1, int(epochs * 0.05))
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epochs), eta_min=1e-6)
    
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[scheduler_warmup, scheduler_cosine], 
        milestones=[warmup_epochs]
    )

    best_val_loss = float('inf')
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
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, progress)
            val_loss = validate(model, val_loader, criterion, device, epoch, progress)
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            history['train'].append(train_loss)
            history['val'].append(val_loss)
            history['lr'].append(current_lr)
            
            console.print(f"[bold cyan]Epoch {epoch}/{epochs}[/bold cyan] | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir / "best_model.pth")
            
            progress.update(epoch_task, advance=1)
            
            if auto_stop_enabled and val_loss <= min_val_loss_threshold:
                console.print(f"\n[bold yellow]验证集 Loss ({val_loss:.4f}) 已达到设定的停止阈值 ({min_val_loss_threshold})，提前终止训练。[/bold yellow]")
                break
                
        torch.save(model.state_dict(), save_dir / "last_model.pth")
        plot_history(history, save_dir / "loss_curve.png")
        
        log_file = save_dir / "train_log.txt"
        with log_file.open("w", encoding="utf-8") as f:
            f.write("Epoch\tLR\tTrain_Loss\tVal_Loss\n")
            for i in range(len(history['train'])):
                f.write(f"{i+1}\t{history['lr'][i]:.6f}\t{history['train'][i]:.6f}\t{history['val'][i]:.6f}\n")
        
        console.print("\n[bold cyan]正在生成识别效果可视化图片...[/bold cyan]")
        model.load_state_dict(torch.load(save_dir / "best_model.pth"))
        
        visualize_predictions(model, train_loader, device, save_dir, prefix="train", progress=progress, input_size=input_size, grid_size=grid_size, num_samples=5)
        visualize_predictions(model, val_loader, device, save_dir, prefix="val", progress=progress, input_size=input_size, grid_size=grid_size, num_samples=5)

    console.print(f"\n[bold green]训练与评估完成！所有结果已保存至: {save_dir.absolute()}[/bold green]")

if __name__ == "__main__":
    main()