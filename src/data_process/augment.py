import cv2
import random
import numpy as np
import shutil
import threading
from pathlib import Path
from queue import Queue

# 引入 rich 组件
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeRemainingColumn, ProgressColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

class MofNCompleteColumn(ProgressColumn):
    """自定义列显示 n/m 格式"""
    def render(self, task):
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        return Text(f"{completed}/{total}", style="progress.remaining")


def process_img(img, brightness_range):
    """
    图像处理：执行随机增强，并记录变换操作。
    返回: (处理后的图像, status 二维列表)
    """
    aug_img = img.copy()
    status = [] 
    
    # 1. 亮度调整
    if random.random() < 0.8:  # 80%概率触发曝光调整
        factor = random.uniform(*brightness_range)
        aug_img = np.clip(aug_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        status.append(['brightness', factor])
        
    # 2. 随机 Resize 
    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        aug_img = cv2.resize(aug_img, None, fx=scale, fy=scale)
        status.append(['resize', scale])
        
    # 3. 旋转处理
    if random.random() < 0.5:
        # Resize 后图像尺寸发生改变，需重新获取长宽计算中心点
        h, w = aug_img.shape[:2]
        angle = random.uniform(-15, 15)
        center = (w / 2, h / 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M, (w, h))
        status.append(['rotate', angle, center])
        
    # cv2.imwrite 期望 BGR 格式，无需进行 RGB 转换
    return aug_img, status


def process_label(label_lines, status, filename=""):
    """
    标签处理：适配 9 个值 (1个类别 + 4个角点) 的数据格式
    """
    new_lines = []
    
    for line in label_lines:
        # 兼容以逗号或多余空格分隔的格式
        clean_line = line.replace(',', ' ').strip()
        parts = clean_line.split()
        
        if not parts:
            continue
            
        # 针对 9个值 的校验 (1个分类ID + 8个坐标值)
        if len(parts) < 9:
            console.print(f"[yellow]跳过异常行 (数量不足9) [{filename}]: '{clean_line}'[/yellow]")
            continue
            
        class_id = parts[0]
        
        try:
            # 提取后续的 8 个坐标值，构建 4x2 矩阵
            pts = np.array([float(x) for x in parts[1:9]]).reshape(-1, 2)
        except ValueError:
            console.print(f"[yellow]跳过异常行 (含非数字字符) [{filename}]: '{clean_line}'[/yellow]")
            continue
            
        for step_idx, op in enumerate(status):
            op_type = op[0]
            
            if op_type == 'resize':
                scale = op[1]
                # 若标签是 0~1 的归一化坐标，请将此行注释掉或加上条件判断
                pts = pts * scale 
                
            elif op_type == 'rotate':
                angle = op[1]
                cx, cy = op[2]
                
                # OpenCV 旋转矩阵数学变换 (角度取负是因为图像坐标系Y轴向下)
                theta = np.radians(-angle) 
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                
                pts = pts - np.array([cx, cy])
                rotated_pts = np.empty_like(pts)
                rotated_pts[:, 0] = pts[:, 0] * cos_t - pts[:, 1] * sin_t
                rotated_pts[:, 1] = pts[:, 0] * sin_t + pts[:, 1] * cos_t
                pts = rotated_pts + np.array([cx, cy])
                
        # 展平矩阵并格式化，保留6位小数
        pts_flat = pts.flatten()
        coords_str = " ".join([f"{coord:.6f}" for coord in pts_flat])
        new_lines.append(f"{class_id} {coords_str}")
        
    return new_lines


def augment_worker(task_queue: Queue, progress: Progress, task_id, brightness_range):
    """
    后台线程：负责图像增强处理及标签同步计算
    """
    while True:
        task = task_queue.get()
        if task is None:
            break
            
        img_path, out_img_path, label_path, out_label_path = task
        
        try:
            # 1. 图像处理 (曝光增强及几何变换)
            img = cv2.imread(str(img_path))
            if img is not None:
                aug_img, status = process_img(img, brightness_range)
                cv2.imwrite(str(out_img_path), aug_img)
            else:
                progress.console.print(f"[red]图像读取失败 {img_path.name}[/red]")
                progress.advance(task_id)
                task_queue.task_done()
                continue
            
            # 2. 标签处理 (根据 status 同步变换)
            if label_path.exists():
                with open(label_path, 'r') as f:
                    label_lines = f.readlines()
                    
                new_labels = process_label(label_lines, status, filename=label_path.name)
                
                # 即使标签为空也写入空白文件，保持图像与标签文件对齐
                with open(out_label_path, 'w') as f:
                    if new_labels:
                        f.write("\n".join(new_labels) + "\n")
                    
                if not new_labels and label_lines:
                     progress.console.print(f"[yellow]警告: {label_path.name} 处理后标签为空[/yellow]")
            else:
                progress.console.print(f"[yellow]未找到标签文件 {label_path.name}[/yellow]")
                
        except Exception as e:
            progress.console.print(f"[red]文件处理异常 {img_path.name}: {e}[/red]")
            
        progress.advance(task_id)
        task_queue.task_done()


def run_augment_pipeline(input_dir: str, output_dir: str, num_workers: int = 8, brightness_range=(0.6, 1.4)):
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists():
        console.print(f"[bold red]错误：[/bold red]找不到输入目录 {in_path}")
        return

    # 1. 扫描数据
    tasks = []
    with console.status("[bold green]正在扫描原始数据..."):
        # 遍历类别目录
        for class_dir in in_path.iterdir():
            if not class_dir.is_dir(): continue
            
            class_id = class_dir.name
            photos_dir = class_dir / "photos"
            labels_dir = class_dir / "labels"

            if not photos_dir.exists(): continue

            for img_file in photos_dir.glob("*.jpg"):
                label_file = labels_dir / (img_file.stem + ".txt")
                
                # 定义输出路径
                out_class_dir = out_path / class_id
                out_img = out_class_dir / "photos" / f"aug_{img_file.name}"
                out_lab = out_class_dir / "labels" / f"aug_{label_file.name}"
                
                tasks.append((img_file, out_img, label_file, out_lab))

    if not tasks:
        console.print("[yellow]未发现可处理的图片数据。[/yellow]")
        return

    # 2. 刷新输出文件夹
    if out_path.exists():
        with console.status(f"[bold red]正在清理旧数据: {output_dir}..."):
            shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # 3. 策略预览
    table = Table(title="数据增强流水线", header_style="bold cyan")
    table.add_column("输入目录", style="dim")
    table.add_column("输出目录")
    table.add_column("总任务数", justify="right", style="green")
    table.add_column("增强策略", justify="center")
    table.add_row(input_dir, output_dir, str(len(tasks)), "亮度/缩放/旋转")
    console.print(table)

    # 4. 执行多线程任务
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console
    )

    with progress:
        main_task = progress.add_task("[magenta]混合增强处理中...", total=len(tasks))
        task_queue = Queue(maxsize=2000)
        
        # 启动 Worker
        threads = []
        for _ in range(num_workers):
            t = threading.Thread(
                target=augment_worker, 
                args=(task_queue, progress, main_task, brightness_range),
                daemon=True
            )
            t.start()
            threads.append(t)

        # 准备输出子文件夹并分发任务
        created_dirs = set()
        for t in tasks:
            out_img_parent = t[1].parent
            out_lab_parent = t[3].parent
            
            # 避免重复创建目录
            if out_img_parent not in created_dirs:
                out_img_parent.mkdir(parents=True, exist_ok=True)
                out_lab_parent.mkdir(parents=True, exist_ok=True)
                created_dirs.add(out_img_parent)
            
            task_queue.put(t)

        # 停止信号
        for _ in range(num_workers):
            task_queue.put(None)
        
        for t in threads:
            t.join()

    console.print(Panel(
        f"✅ [bold green]增强处理完成！[/bold green]\n\n数据路径: [underline]{out_path.absolute()}[/underline]\n处理总数: {len(tasks)}", 
        border_style="green",
        title="Success"
    ))

if __name__ == "__main__":
    run_augment_pipeline(
        input_dir="./data/balance", 
        output_dir="./data/augment", 
        num_workers=8, 
        brightness_range=(0.6, 1.4)
    )