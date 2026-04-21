import cv2
import random
import numpy as np
import shutil
import threading
import copy
from pathlib import Path
from queue import Queue
from dataclasses import dataclass
from typing import Tuple, List
import yaml

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TaskProgressColumn, TimeRemainingColumn, ProgressColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

@dataclass
class AugmentConfig:
    """图像增强配置参数类"""
    # 基础与光学
    brightness_prob: float = 0.8
    brightness_range: Tuple[float, float] = (0.6, 1.4)
    blur_prob: float = 0.3
    blur_ksize: List[int] = None # type: ignore
    hsv_prob: float = 0.5
    hsv_h_gain: float = 0.015
    hsv_s_gain: float = 0.4
    hsv_v_gain: float = 0.4
    noise_prob: float = 0.3
    bloom_prob: float = 0.3

    # 几何变换
    flip_prob: float = 0.5
    scale_prob: float = 0.5
    scale_range: Tuple[float, float] = (0.8, 1.2)
    rotate_prob: float = 0.5
    rotate_range: Tuple[float, float] = (-15, 15)
    translate_prob: float = 0.4
    translate_range: float = 0.1
    perspective_prob: float = 0.3
    perspective_factor: float = 0.08
    
    # 背景替换
    bg_replace_prob: float = 0.3          
    bg_dir: str = "./background"          
    bg_radius_factor: float = 1.3         

    # 遮挡
    occ_prob: float = 0.5
    occ_radius_pct: float = 0.2
    occ_size_pct: Tuple[float, float] = (0.05, 0.15)
    vis_heavy_threshold: float = 0.7
    vis_part_threshold: float = 0.1

    def __post_init__(self):
        if self.blur_ksize is None:
            self.blur_ksize = [3, 5, 7]

    @staticmethod
    def from_yaml(yaml_path: str):
        cfg = AugmentConfig()
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                aug_data = data.get('kielas_rm_train', {}).get('dataset', {}).get('augment', {})
                
                if aug_data:
                    for key, value in aug_data.items():
                        if hasattr(cfg, key):
                            if isinstance(value, list) and len(value) == 2 and key != 'blur_ksize':
                                value = tuple(value)
                            setattr(cfg, key, value)
                    console.print(f"[green]已从 {yaml_path} 加载增强配置[/green]")
                else:
                    console.print("[yellow]YAML 中未发现 augment 配置，使用默认参数[/yellow]")
        except Exception as e:
            console.print(f"[red]加载 YAML 失败: {e}，将使用默认参数[/red]")
        return cfg

class MofNCompleteColumn(ProgressColumn):
    def render(self, task):
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        return Text(f"{completed}/{total}", style="progress.remaining")

def parse_labels(label_lines, filename=""):
    parsed = []
    for line in label_lines:
        clean_line = line.replace(',', ' ').strip()
        parts = clean_line.split()
        if not parts or len(parts) < 9: continue
        class_id = parts[0]
        try:
            if len(parts) == 9:
                visibility = 2
                pts = np.array([float(x) for x in parts[1:9]]).reshape(-1, 2)
            else:
                visibility = int(float(parts[1]))
                pts = np.array([float(x) for x in parts[2:10]]).reshape(-1, 2)
            parsed.append({'class_id': class_id, 'vis': visibility, 'pts': pts})
        except ValueError:
            continue
    return parsed

def format_labels(labels):
    new_lines = []
    for lab in labels:
        pts_flat = lab['pts'].flatten()
        coords_str = " ".join([f"{coord:.6f}" for coord in pts_flat])
        new_lines.append(f"{lab['class_id']} {lab['vis']} {coords_str}")
    return new_lines

def process_data(img, labels, cfg: AugmentConfig, bg_paths: list = None): # type: ignore
    aug_img = img.copy()
    aug_labels = copy.deepcopy(labels)
    h, w = aug_img.shape[:2]
        
    # ================= 1. 光学与色彩空间变换 =================
    
    if random.random() < cfg.brightness_prob:
        factor = random.uniform(*cfg.brightness_range)
        aug_img = np.clip(aug_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if random.random() < cfg.blur_prob:
        ksize = random.choice(cfg.blur_ksize)
        angle = random.uniform(0, 180)
        M_blur = cv2.getRotationMatrix2D((ksize/2, ksize/2), angle, 1)
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize-1)/2), :] = np.ones(ksize)
        kernel = cv2.warpAffine(kernel, M_blur, (ksize, ksize))
        kernel = kernel / ksize
        aug_img = cv2.filter2D(aug_img, -1, kernel)

    if random.random() < cfg.hsv_prob:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_gain = random.uniform(-cfg.hsv_h_gain, cfg.hsv_h_gain) * 180
        s_gain = random.uniform(-cfg.hsv_s_gain, cfg.hsv_s_gain)
        v_gain = random.uniform(-cfg.hsv_v_gain, cfg.hsv_v_gain)
        hsv[:, :, 0] = (hsv[:, :, 0] + h_gain) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + s_gain), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + v_gain), 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < cfg.noise_prob:
        noise = np.random.normal(0, 15, aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if aug_labels and random.random() < cfg.bloom_prob:
        bloom_layer = np.zeros_like(aug_img, dtype=np.float32)
        for lab in aug_labels:
            if random.random() < 0.5: 
                center = np.mean(lab['pts'], axis=0).astype(int)
                cv2.circle(bloom_layer, tuple(center), int(min(h, w) * 0.02), (255, 255, 255), -1)
        bloom_layer = cv2.GaussianBlur(bloom_layer, (0, 0), sigmaX=15)
        aug_img = np.clip(aug_img.astype(np.float32) + bloom_layer, 0, 255).astype(np.uint8)

    # ================= 2. 高级几何变换 =================

    if random.random() < cfg.flip_prob:
        aug_img = cv2.flip(aug_img, 1)
        for lab in aug_labels:
            lab['pts'][:, 0] = w - lab['pts'][:, 0]
            old_pts = lab['pts'].copy()
            # 正确的映射：左边变右边，右边变左边，上下保持不变
            lab['pts'][0] = old_pts[2]  # 新左下 = 旧右下
            lab['pts'][1] = old_pts[3]  # 新左上 = 旧右上
            lab['pts'][2] = old_pts[0]  # 新右下 = 旧左下
            lab['pts'][3] = old_pts[1]  # 新右上 = 旧左上  

    if random.random() < cfg.scale_prob:
        scale = random.uniform(*cfg.scale_range)
        aug_img = cv2.resize(aug_img, None, fx=scale, fy=scale)
        for lab in aug_labels:
            lab['pts'] = lab['pts'] * scale
        h, w = aug_img.shape[:2]
            
    if random.random() < cfg.rotate_prob:
        angle = random.uniform(*cfg.rotate_range)
        center = (w / 2, h / 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_img = cv2.warpAffine(aug_img, M_rot, (w, h))
        
        theta = np.radians(-angle) 
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for lab in aug_labels:
            pts = lab['pts'] - np.array(center)
            rotated_pts = np.empty_like(pts)
            rotated_pts[:, 0] = pts[:, 0] * cos_t - pts[:, 1] * sin_t
            rotated_pts[:, 1] = pts[:, 0] * sin_t + pts[:, 1] * cos_t
            lab['pts'] = rotated_pts + np.array(center)

    if random.random() < cfg.translate_prob:
        tx = random.uniform(-cfg.translate_range, cfg.translate_range) * w
        ty = random.uniform(-cfg.translate_range, cfg.translate_range) * h
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]]) # type: ignore
        aug_img = cv2.warpAffine(aug_img, M_trans, (w, h), borderValue=(0,0,0)) # type: ignore
        for lab in aug_labels:
            lab['pts'][:, 0] += tx
            lab['pts'][:, 1] += ty

    if random.random() < cfg.perspective_prob:
        margin = min(h, w) * cfg.perspective_factor
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]]) # type: ignore
        pts2 = np.float32([
            [random.uniform(-margin, margin), random.uniform(-margin, margin)],
            [w + random.uniform(-margin, margin), random.uniform(-margin, margin)],
            [random.uniform(-margin, margin), h + random.uniform(-margin, margin)],
            [w + random.uniform(-margin, margin), h + random.uniform(-margin, margin)]
        ]) # type: ignore
        M_persp = cv2.getPerspectiveTransform(pts1, pts2) # type: ignore
        aug_img = cv2.warpPerspective(aug_img, M_persp, (w, h), borderValue=(0,0,0))
        
        for lab in aug_labels:
            pts_reshaped = np.array([lab['pts']], dtype=np.float32)
            lab['pts'] = cv2.perspectiveTransform(pts_reshaped, M_persp)[0]

    # ================= 3. 边界检查与可见度修正 =================
    for lab in aug_labels:
        out_count = sum(1 for pt in lab['pts'] if pt[0] < 0 or pt[0] >= w or pt[1] < 0 or pt[1] >= h)
        
        if out_count >= 3:  
            lab['vis'] = 0
        elif out_count > 0: 
            lab['vis'] = min(lab['vis'], 1)
            
        center_x, center_y = np.mean(lab['pts'], axis=0)
        if center_x < 0 or center_x >= w or center_y < 0 or center_y >= h:
            lab['vis'] = 0

    # ================= 4. 随机背景替换 =================
    if aug_labels and bg_paths and random.random() < cfg.bg_replace_prob:
        bg_path = random.choice(bg_paths)
        bg_img = cv2.imread(str(bg_path))
        
        if bg_img is not None:
            bg_img = cv2.resize(bg_img, (w, h))
            mask = np.zeros((h, w), dtype=np.float32)
            
            all_radii = []
            
            for lab in aug_labels:
                if lab['vis'] > 0:
                    pts = lab['pts']
                    center = np.mean(pts, axis=0)
                    distances = np.linalg.norm(pts - center, axis=1)
                    base_radius = np.max(distances)
                    radius = int(base_radius * cfg.bg_radius_factor)
                    all_radii.append(radius)
                    cv2.circle(mask, (int(center[0]), int(center[1])), radius, 1.0, -1)
            
            if all_radii:
                gray_aug = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                roi_pixels = gray_aug[mask > 0.5]
                
                if roi_pixels.size > 0:
                    roi_mean = np.mean(roi_pixels)
                    gray_bg = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    bg_mean = np.mean(gray_bg)
                    
                    brightness_gain = np.clip(roi_mean / (bg_mean + 1e-6), 0.5, 1.5)
                    bg_img = np.clip(bg_img.astype(np.float32) * brightness_gain, 0, 255).astype(np.uint8)

                avg_radius = np.mean(all_radii)
                dynamic_ksize = int(avg_radius * 0.2)
                if dynamic_ksize % 2 == 0:
                    dynamic_ksize += 1
                dynamic_ksize = max(3, dynamic_ksize)

                mask = cv2.GaussianBlur(mask, (dynamic_ksize, dynamic_ksize), 0)
                
                mask_3d = np.expand_dims(mask, axis=-1)
                aug_img = (aug_img.astype(np.float32) * mask_3d + 
                           bg_img.astype(np.float32) * (1 - mask_3d)).astype(np.uint8)

    # ================= 5. 随机遮挡 =================
    if aug_labels and random.random() < cfg.occ_prob:
        radius = (w * cfg.occ_radius_pct) / 2.0
        occ_boxes = []
        
        for lab in aug_labels:
            if random.random() < 0.5:
                tx, ty = np.mean(lab['pts'], axis=0)
                angle = random.uniform(0, 2 * np.pi)
                dist = random.uniform(0, radius)
                cx, cy = tx + dist * np.cos(angle), ty + dist * np.sin(angle)
                
                occ_w = int(w * random.uniform(*cfg.occ_size_pct))
                occ_h = int(h * random.uniform(*cfg.occ_size_pct))
                
                occ_x1, occ_y1 = int(cx - occ_w / 2), int(cy - occ_h / 2)
                occ_x2, occ_y2 = occ_x1 + occ_w, occ_y1 + occ_h
                occ_boxes.append((occ_x1, occ_y1, occ_x2, occ_y2))
                
                draw_y1, draw_y2 = max(0, occ_y1), min(h, occ_y2)
                draw_x1, draw_x2 = max(0, occ_x1), min(w, occ_x2)
                if draw_y1 < draw_y2 and draw_x1 < draw_x2:
                    aug_img[draw_y1:draw_y2, draw_x1:draw_x2] = 0
        
        if occ_boxes:
            for lab in aug_labels:
                covered_points = 0
                for pt in lab['pts']:
                    px, py = pt[0], pt[1]
                    if any(x1 <= px <= x2 and y1 <= py <= y2 for x1, y1, x2, y2 in occ_boxes):
                        covered_points += 1

                pts = lab['pts']
                min_x, min_y = np.min(pts, axis=0)
                max_x, max_y = np.max(pts, axis=0)
                target_area = (max_x - min_x) * (max_y - min_y)
                
                is_heavily_occluded = False
                is_partially_occluded = False

                if target_area > 0:
                    for x1, y1, x2, y2 in occ_boxes:
                        ix1, iy1 = max(min_x, x1), max(min_y, y1)
                        ix2, iy2 = min(max_x, x2), min(max_y, y2)
                        if ix1 < ix2 and iy1 < iy2:
                            overlap_ratio = ((ix2 - ix1) * (iy2 - iy1)) / target_area
                            if overlap_ratio > cfg.vis_heavy_threshold:
                                is_heavily_occluded = True
                            elif overlap_ratio > cfg.vis_part_threshold:
                                is_partially_occluded = True

                if covered_points == 4 or is_heavily_occluded:
                    lab['vis'] = 0
                elif covered_points > 0 or is_partially_occluded:
                    lab['vis'] = min(lab['vis'], 1)
        
    return aug_img, aug_labels

def augment_worker(task_queue: Queue, progress: Progress, task_id, cfg: AugmentConfig, bg_paths: list):
    while True:
        task = task_queue.get()
        if task is None: break
        img_path, out_img_path, label_path, out_label_path = task
        try:
            parsed_labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    parsed_labels = parse_labels(f.readlines(), filename=label_path.name)
            
            img = cv2.imread(str(img_path))
            if img is None:
                progress.advance(task_id)
                task_queue.task_done()
                continue
                
            aug_img, aug_labels = process_data(img, parsed_labels, cfg, bg_paths)
            cv2.imwrite(str(out_img_path), aug_img)
            
            new_label_lines = format_labels(aug_labels)
            with open(out_label_path, 'w') as f:
                if new_label_lines:
                    f.write("\n".join(new_label_lines) + "\n")
        except Exception as e:
            progress.console.print(f"[red]处理异常 {img_path.name}: {e}[/red]")
        progress.advance(task_id)
        task_queue.task_done()

def run_augment_pipeline(dataset_dir: str, num_workers: int = 8, cfg: AugmentConfig = None): # type: ignore
    if cfg is None: cfg = AugmentConfig()
    base_path = Path(dataset_dir)
    
    # 强制定位到拆分后的训练集目录
    train_images_dir = base_path / "images" / "train"
    train_labels_dir = base_path / "labels" / "train"

    if not train_images_dir.exists() or not train_labels_dir.exists():
        console.print(f"[bold red]错误：[/bold red]找不到训练集目录 {train_images_dir}")
        return

    tasks = []
    with console.status("[bold green]正在扫描训练集原始数据..."):
        for img_file in train_images_dir.glob("*.*"):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']: continue
            
            # 过滤掉已经增强过的数据，防止二次运行导致指数级膨胀
            if img_file.name.startswith("aug_"): continue 

            label_file = train_labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists(): continue
            
            # 构造同级目录下的输出路径
            out_img_file = train_images_dir / f"aug_{img_file.name}"
            out_label_file = train_labels_dir / f"aug_{label_file.name}"
            
            tasks.append((img_file, out_img_file, label_file, out_label_file))

    if not tasks: 
        console.print("[yellow]未扫描到需要增强的数据或已全部增强完毕。[/yellow]")
        return

    bg_paths = []
    bg_dir = Path(cfg.bg_dir)
    if bg_dir.exists():
        bg_paths = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png")) + list(bg_dir.glob("*.jpeg"))
        if bg_paths:
            console.print(f"[green]已加载 {len(bg_paths)} 张背景图片用于替换增强[/green]")
        else:
            console.print("[yellow]背景文件夹为空，将跳过背景替换增强[/yellow]")
    else:
        console.print(f"[yellow]未找到背景文件夹 {cfg.bg_dir}，将跳过背景替换增强[/yellow]")

    # 打印配置信息
    table = Table(title="定向训练集数据增强", header_style="bold cyan")
    table.add_column("模块")
    table.add_column("配置与状态")
    table.add_row("基础架构", f"目标目录:{train_images_dir} | 线程:{num_workers}")
    table.add_row("光学增强", f"模糊:{cfg.blur_prob} | HSV抖动:{cfg.hsv_prob} | 噪声:{cfg.noise_prob} | 光晕:{cfg.bloom_prob}")
    table.add_row("几何变换", f"平移:{cfg.translate_prob} | 缩放:{cfg.scale_prob} | 翻转:{cfg.flip_prob} | 透视:{cfg.perspective_prob}")
    table.add_row("高级遮挡", f"背景替换:{cfg.bg_replace_prob} (半径倍数:{cfg.bg_radius_factor}) | 随机遮挡:{cfg.occ_prob}")
    console.print(table)

    progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                        BarColumn(bar_width=40), TaskProgressColumn(), MofNCompleteColumn(),
                        TimeRemainingColumn(), console=console)

    with progress:
        main_task = progress.add_task("[magenta]执行定向增强处理...", total=len(tasks))
        task_queue = Queue(maxsize=2000)
        
        threads = [threading.Thread(target=augment_worker, args=(task_queue, progress, main_task, cfg, bg_paths), daemon=True) 
                   for _ in range(num_workers)]
        for t in threads: t.start()

        for t in tasks:
            task_queue.put(t)

        for _ in range(num_workers): task_queue.put(None)
        for t in threads: t.join()

    console.print(Panel(f"✅ 训练集增强完毕！共生成 {len(tasks)} 张增强样本，验证集保持纯净。", border_style="green", title="Success"))

if __name__ == "__main__":
    config_path = "config.yaml"
    config = AugmentConfig.from_yaml(config_path)

    run_augment_pipeline(
        dataset_dir="./data/datasets", 
        num_workers=8,
        cfg=config
    )