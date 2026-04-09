import cv2
import random
import numpy as np
import shutil
import threading
import copy
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


def parse_labels(label_lines, filename=""):
    """
    解析标签，向下兼容 9个值 和 10个值。
    返回结构化字典列表。
    """
    parsed = []
    for line in label_lines:
        clean_line = line.replace(',', ' ').strip()
        parts = clean_line.split()
        
        if not parts or len(parts) < 9:
            continue
            
        class_id = parts[0]
        try:
            if len(parts) == 9:
                visibility = 2
                pts = np.array([float(x) for x in parts[1:9]]).reshape(-1, 2)
            else:
                visibility = int(float(parts[1]))
                pts = np.array([float(x) for x in parts[2:10]]).reshape(-1, 2)
            
            parsed.append({
                'class_id': class_id, 
                'vis': visibility, 
                'pts': pts
            })
        except ValueError:
            continue
    return parsed


def format_labels(labels):
    """
    将结构化的标签转回字符串列表以供写入文件
    """
    new_lines = []
    for lab in labels:
        pts_flat = lab['pts'].flatten()
        coords_str = " ".join([f"{coord:.6f}" for coord in pts_flat])
        new_lines.append(f"{lab['class_id']} {lab['vis']} {coords_str}")
    return new_lines


def process_data(img, labels, brightness_range, occ_radius_pct=0.2):
    """
    同步处理图像与标签增强。
    occ_radius_pct: 遮挡半径占图像宽度的百分比。
    """
    aug_img = img.copy()
    # 深拷贝避免修改原始数据
    aug_labels = copy.deepcopy(labels) 
    
    # 1. 亮度调整 (不影响标签坐标)
    if random.random() < 0.8:
        factor = random.uniform(*brightness_range)
        aug_img = np.clip(aug_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        
    # 2. 随机 Resize 
    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        aug_img = cv2.resize(aug_img, None, fx=scale, fy=scale)
        for lab in aug_labels:
            lab['pts'] = lab['pts'] * scale
            
    # 3. 旋转处理
    if random.random() < 0.5:
        h, w = aug_img.shape[:2]
        angle = random.uniform(-15, 15)
        center = (w / 2, h / 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M, (w, h))
        
        # 同步旋转坐标点
        theta = np.radians(-angle) 
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        for lab in aug_labels:
            pts = lab['pts'] - np.array(center)
            rotated_pts = np.empty_like(pts)
            rotated_pts[:, 0] = pts[:, 0] * cos_t - pts[:, 1] * sin_t
            rotated_pts[:, 1] = pts[:, 0] * sin_t + pts[:, 1] * cos_t
            lab['pts'] = rotated_pts + np.array(center)

    # 4. 围绕目标的随机遮挡 (Cutout)
    if aug_labels and random.random() < 0.5:
        h, w = aug_img.shape[:2]
        # 半径: (图像x轴长度 * 百分比) / 2
        radius = (w * occ_radius_pct) / 2.0
        
        occ_boxes = []
        
        for lab in aug_labels:
            # 每个目标有 50% 概率触发周围遮挡
            if random.random() < 0.5:
                # 获取当前变换后的目标中心点
                tx, ty = np.mean(lab['pts'], axis=0)
                
                # 在半径范围内随机取点作为遮挡块中心
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(0, radius)
                cx = tx + dist * np.cos(angle)
                cy = ty + dist * np.sin(angle)
                
                # 随机生成遮挡块宽高 (例如图像宽高的 5% 到 15%)
                occ_w = int(w * random.uniform(0.05, 0.15))
                occ_h = int(h * random.uniform(0.05, 0.15))
                
                # 遮挡块左上角和右下角
                occ_x1 = int(cx - occ_w / 2)
                occ_y1 = int(cy - occ_h / 2)
                occ_x2 = occ_x1 + occ_w
                occ_y2 = occ_y1 + occ_h
                
                occ_boxes.append((occ_x1, occ_y1, occ_x2, occ_y2))
                
                # 在图像上绘制黑色遮挡块 (注意截断防越界)
                draw_y1, draw_y2 = max(0, occ_y1), min(h, occ_y2)
                draw_x1, draw_x2 = max(0, occ_x1), min(w, occ_x2)
                if draw_y1 < draw_y2 and draw_x1 < draw_x2:
                    aug_img[draw_y1:draw_y2, draw_x1:draw_x2] = 0
        
        # 统一计算遮挡对可见度的影响
        if occ_boxes:
            for lab in aug_labels:
                covered_points = 0
                for pt in lab['pts']:
                    px, py = pt[0], pt[1]
                    # 判断当前关键点是否落在任意一个遮挡矩形框内
                    if any(x1 <= px <= x2 and y1 <= py <= y2 for x1, y1, x2, y2 in occ_boxes):
                        covered_points += 1
                
                if covered_points == 4:
                    lab['vis'] = 0
                elif covered_points > 0:
                    lab['vis'] = min(lab['vis'], 1)
        
    return aug_img, aug_labels


def augment_worker(task_queue: Queue, progress: Progress, task_id, brightness_range, occ_radius_pct):
    """
    后台线程：负责读取数据并执行同步增强处理
    """
    while True:
        task = task_queue.get()
        if task is None:
            break
            
        img_path, out_img_path, label_path, out_label_path = task
        
        try:
            # 1. 读取标签文件
            parsed_labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    label_lines = f.readlines()
                parsed_labels = parse_labels(label_lines, filename=label_path.name)
            else:
                progress.console.print(f"[yellow]未找到标签文件 {label_path.name}[/yellow]")

            # 2. 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                progress.console.print(f"[red]图像读取失败 {img_path.name}[/red]")
                progress.advance(task_id)
                task_queue.task_done()
                continue
                
            # 3. 同步执行增强处理
            aug_img, aug_labels = process_data(
                img, 
                parsed_labels, 
                brightness_range, 
                occ_radius_pct
            )
            
            # 4. 保存增强后的图像
            cv2.imwrite(str(out_img_path), aug_img)
            
            # 5. 保存增强后的标签
            new_label_lines = format_labels(aug_labels)
            with open(out_label_path, 'w') as f:
                if new_label_lines:
                    f.write("\n".join(new_label_lines) + "\n")
                    
            if not new_label_lines and label_path.exists():
                 progress.console.print(f"[yellow]警告: {label_path.name} 处理后无有效标签[/yellow]")
                
        except Exception as e:
            progress.console.print(f"[red]文件处理异常 {img_path.name}: {e}[/red]")
            
        progress.advance(task_id)
        task_queue.task_done()


def generate_yaml(output_dir: Path):
    """扫描输出目录并生成 train.yaml 配置文件"""
    yaml_path = output_dir / "train.yaml"
    
    class_counts = {}
    for class_dir in output_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        labels_dir = class_dir / "labels"
        if labels_dir.exists():
            count = len(list(labels_dir.glob("*.txt")))
            if count > 0:
                class_counts[class_dir.name] = count

    if not class_counts:
        console.print("[yellow]未发现有效标签数据，跳过生成 yaml。[/yellow]")
        return

    max_count = max(class_counts.values())
    class_weights = {cid: max_count / count for cid, count in class_counts.items()}
    sorted_cids = sorted(class_counts.keys(), key=lambda x: int(x) if x.isdigit() else x)

    yaml_content = f"# Train Dataset Configuration\n"
    yaml_content += f"path: {output_dir.absolute()}\n"
    yaml_content += f"train: ./\n"
    yaml_content += f"val: ./\n\n"
    yaml_content += f"nc: {len(class_counts)}\n\n"
    
    yaml_content += "names:\n"
    for cid in sorted_cids:
        yaml_content += f"  {cid}: '{cid}'\n"

    yaml_content += "\nweights:\n"
    for cid in sorted_cids:
        yaml_content += f"  {cid}: {class_weights[cid]:.4f}\n"

    yaml_content += """
features_description:
  - class_id
  - visibility
  - left_light_down_x
  - left_light_down_y
  - left_light_up_x
  - left_light_up_y
  - right_light_down_x
  - right_light_down_y
  - right_light_up_x
  - right_light_up_y
"""

    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


def run_augment_pipeline(input_dir: str, output_dir: str, num_workers: int = 8, brightness_range=(0.6, 1.4), occ_radius_pct=0.2):
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists():
        console.print(f"[bold red]错误：[/bold red]找不到输入目录 {in_path}")
        return

    tasks = []
    with console.status("[bold green]正在扫描原始数据..."):
        for class_dir in in_path.iterdir():
            if not class_dir.is_dir(): continue
            
            class_id = class_dir.name
            photos_dir = class_dir / "photos"
            labels_dir = class_dir / "labels"

            if not photos_dir.exists(): continue

            for img_file in photos_dir.glob("*.jpg"):
                label_file = labels_dir / (img_file.stem + ".txt")
                
                out_class_dir = out_path / class_id
                out_img = out_class_dir / "photos" / f"aug_{img_file.name}"
                out_lab = out_class_dir / "labels" / f"aug_{label_file.name}"
                
                tasks.append((img_file, out_img, label_file, out_lab))

    if not tasks:
        console.print("[yellow]未发现可处理的图片数据。[/yellow]")
        return

    if out_path.exists():
        with console.status(f"[bold red]正在清理旧数据: {output_dir}..."):
            shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    table = Table(title="数据增强流水线", header_style="bold cyan")
    table.add_column("输入目录", style="dim")
    table.add_column("输出目录")
    table.add_column("总任务数", justify="right", style="green")
    table.add_column("增强策略", justify="center")
    table.add_row(input_dir, output_dir, str(len(tasks)), "亮度/缩放/旋转/目标周围遮挡")
    console.print(table)

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
        
        threads = []
        for _ in range(num_workers):
            t = threading.Thread(
                target=augment_worker, 
                args=(task_queue, progress, main_task, brightness_range, occ_radius_pct),
                daemon=True
            )
            t.start()
            threads.append(t)

        created_dirs = set()
        for t in tasks:
            out_img_parent = t[1].parent
            out_lab_parent = t[3].parent
            
            if out_img_parent not in created_dirs:
                out_img_parent.mkdir(parents=True, exist_ok=True)
                out_lab_parent.mkdir(parents=True, exist_ok=True)
                created_dirs.add(out_img_parent)
            
            task_queue.put(t)

        for _ in range(num_workers):
            task_queue.put(None)
        
        for t in threads:
            t.join()

    generate_yaml(out_path)

    console.print(Panel(
        f"✅ [bold green]增强处理完成！[/bold green]\n\n数据路径: [underline]{out_path.absolute()}[/underline]\n配置文件: [cyan]{(out_path / 'train.yaml').absolute()}[/cyan]\n处理总数: {len(tasks)}", 
        border_style="green",
        title="Success"
    ))

if __name__ == "__main__":
    run_augment_pipeline(
        input_dir="./data/balance", 
        output_dir="./data/augment", 
        num_workers=8, 
        brightness_range=(0.6, 1.4),
        occ_radius_pct=0.2  # 图像x轴长度的 20% 作为直径占比，即 10% 作为半径（可根据需求调整）
    )