import cv2
import random
import numpy as np
import copy
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
import torch
import torchvision.transforms.functional as TF

@dataclass
class AugmentConfig:
    # --- 基础与光学增强 (迁移至 GPU) ---
    brightness_prob: float = 0.9
    brightness_range: Tuple[float, float] = (0.2, 3.5)
    blur_prob: float = 0.7
    blur_ksize: List[int] = None # type: ignore
    hsv_prob: float = 0.8
    hsv_h_gain: float = 0.010
    hsv_s_gain: float = 0.7
    hsv_v_gain: float = 0.8
    noise_prob: float = 0.7
    bloom_prob: float = 0.8
    bloom_prob_area_range: Tuple[float, float] = (0.01, 0.13) 

    # --- 几何变换增强 (保留在 CPU) ---
    flip_prob: float = 0.5
    scale_prob: float = 0.9
    scale_range: Tuple[float, float] = (0.3, 2.5) 
    rotate_prob: float = 0.8
    rotate_range: Tuple[float, float] = (-45, 45)
    translate_prob: float = 0.8
    translate_range: float = 0.4
    perspective_prob: float = 0.8
    perspective_factor: float = 0.35

    # --- 背景与遮挡 (保留在 CPU) ---
    bg_replace_prob: float = 0.85
    bg_dir: str = "./background"
    
    roi_h_exp: float = 2.2  
    roi_w_exp: float = 1.1  

    occ_prob: float = 0.8
    occ_radius_pct: float = 0.4
    occ_size_pct: Tuple[float, float] = (0.02, 0.35)
    
    def __post_init__(self):
        if self.blur_ksize is None:
            self.blur_ksize = [3, 5, 7, 9, 11]

def get_expanded_roi(pts, h_exp, w_exp):
    """基于两根灯条的向量方向，向外拉伸多边形以覆盖完整装甲板"""
    p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
    
    cl = (p0 + p1) / 2.0
    vl = p1 - p0  
    cr = (p2 + p3) / 2.0
    vr = p3 - p2  

    vw = cr - cl
    W = np.linalg.norm(vw)
    dw = vw / W if W > 0 else np.array([1.0, 0.0], dtype=np.float32)

    p1_new = cl + (vl / 2.0) * h_exp
    p0_new = cl - (vl / 2.0) * h_exp
    p3_new = cr + (vr / 2.0) * h_exp
    p2_new = cr - (vr / 2.0) * h_exp

    offset = dw * W * (w_exp - 1.0) / 2.0
    p1_new -= offset
    p0_new -= offset
    p3_new += offset
    p2_new += offset

    return np.array([p0_new, p1_new, p3_new, p2_new], dtype=np.int32)

def generate_composite_bg(bg_paths, w, h):
    """生成复合背景"""
    if not bg_paths: 
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        grid_size = max(w, h) // 15
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                if (x // grid_size + y // grid_size) % 2 == 0:
                    bg[y:y+grid_size, x:x+grid_size] = (100, 50, 150)
                else:
                    bg[y:y+grid_size, x:x+grid_size] = (50, 150, 50) 
        noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
        return cv2.add(bg, noise)

    bg_path = str(random.choice(bg_paths))
    bg = cv2.imread(bg_path)
    if bg is None: 
        bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg = cv2.resize(bg, (w, h))

    if random.random() < 0.6:
        for _ in range(random.randint(1, 2)):
            patch_path = str(random.choice(bg_paths))
            patch = cv2.imread(patch_path)
            if patch is None: continue
            pw, ph = random.randint(int(w*0.3), int(w*0.7)), random.randint(int(h*0.3), int(h*0.7))
            patch = cv2.resize(patch, (pw, ph))
            px, py = random.randint(0, w - pw), random.randint(0, h - ph)
            bg[py:py+ph, px:px+pw] = patch
    return bg


class AugmentPipeline:
    """数据增强总控流水线：CPU 负责空间逻辑，GPU 负责像素渲染"""
    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    # =========================================================================
    # 供 datasets.py 在 DataLoader 的子进程中调用 (CPU 环境)
    # =========================================================================
    def process_cpu(self, img, labels, bg_paths: list = None):
        """执行所有几何形变、遮挡和背景融合"""
        aug_img = img.copy()
        aug_labels = copy.deepcopy(labels)
        h_orig, w_orig = aug_img.shape[:2]
        
        bg_img = generate_composite_bg(bg_paths, w_orig, h_orig)

        # ================= 1. 核心几何变换与矩阵合并 =================
        if aug_labels:
            if random.random() < self.cfg.flip_prob:
                aug_img = cv2.flip(aug_img, 1)
                for lab in aug_labels:
                    lab['pts'][:, 0] = w_orig - lab['pts'][:, 0]
                    old_pts = lab['pts'].copy()
                    lab['pts'][0], lab['pts'][1] = old_pts[2], old_pts[3] 
                    lab['pts'][2], lab['pts'][3] = old_pts[0], old_pts[1] 

            primary_pts = aug_labels[0]['pts']
            plate_area = max(cv2.contourArea(get_expanded_roi(primary_pts, 1.0, 1.0)), 1.0)
            cx, cy = np.mean(primary_pts, axis=0)

            # --- 阶段 A：纯形状变换 ---
            angle = random.uniform(*self.cfg.rotate_range) if random.random() < self.cfg.rotate_prob else 0.0
            T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
            R_shape = np.vstack([cv2.getRotationMatrix2D((0, 0), angle, 1.0), [0, 0, 1]]).astype(np.float32)
            T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
            M_shape = T2 @ R_shape @ T1

            if random.random() < self.cfg.perspective_prob:
                margin = min(h_orig, w_orig) * self.cfg.perspective_factor
                pts1 = np.float32([[0, 0], [w_orig, 0], [0, h_orig], [w_orig, h_orig]])
                pts2 = np.float32([
                    [random.uniform(0, margin), random.uniform(0, margin)],
                    [w_orig - random.uniform(0, margin), random.uniform(0, margin)],
                    [random.uniform(0, margin), h_orig - random.uniform(0, margin)],
                    [w_orig - random.uniform(0, margin), h_orig - random.uniform(0, margin)]
                ])
                M_persp = cv2.getPerspectiveTransform(pts1, pts2).astype(np.float32)
                M_shape = M_persp @ M_shape 

            # --- 阶段 B：面积补偿 ---
            pts_transformed = cv2.perspectiveTransform(np.array([primary_pts], dtype=np.float32), M_shape)[0]
            new_area = max(cv2.contourArea(get_expanded_roi(pts_transformed, 1.0, 1.0)), 1.0)
            new_cx, new_cy = np.mean(pts_transformed, axis=0)

            if random.random() < self.cfg.scale_prob:
                target_area_ratio = random.uniform(*self.cfg.scale_range)
                target_area = w_orig * h_orig * target_area_ratio
                final_scale = np.sqrt(target_area / new_area)
            else:
                final_scale = np.sqrt(plate_area / new_area)

            # --- 阶段 C：平移变换 ---
            tx, ty = 0.0, 0.0
            if random.random() < self.cfg.translate_prob:
                dt_x, dt_y = self.cfg.translate_range * w_orig, self.cfg.translate_range * h_orig
                tx = random.uniform(-dt_x, dt_x)
                ty = random.uniform(-dt_y, dt_y)
                ncx, ncy = np.clip(new_cx + tx, w_orig*0.1, w_orig*0.9), np.clip(new_cy + ty, h_orig*0.1, h_orig*0.9)
                tx, ty = ncx - new_cx, ncy - new_cy

            # --- 阶段 D：矩阵组装 ---
            T_scale_back = np.array([[1, 0, -new_cx], [0, 1, -new_cy], [0, 0, 1]], dtype=np.float32)
            S_mat = np.array([[final_scale, 0, 0], [0, final_scale, 0], [0, 0, 1]], dtype=np.float32)
            T_scale_fwd = np.array([[1, 0, new_cx + tx], [0, 1, new_cy + ty], [0, 0, 1]], dtype=np.float32)
            M_total = (T_scale_fwd @ S_mat @ T_scale_back) @ M_shape

            if random.random() < self.cfg.perspective_prob:
                margin = min(h_orig, w_orig) * self.cfg.perspective_factor
                pts1 = np.float32([[0, 0], [w_orig, 0], [0, h_orig], [w_orig, h_orig]])
                pts2 = np.float32([
                    [random.uniform(0, margin), random.uniform(0, margin)],
                    [w_orig - random.uniform(0, margin), random.uniform(0, margin)],
                    [random.uniform(0, margin), h_orig - random.uniform(0, margin)],
                    [w_orig - random.uniform(0, margin), h_orig - random.uniform(0, margin)]
                ])
                M_persp = cv2.getPerspectiveTransform(pts1, pts2).astype(np.float32)
                M_total = M_persp @ M_total 

            # ================= 2. 极速安全边界校验 =================
            all_pts = np.vstack([cv2.perspectiveTransform(np.array([lab['pts']], dtype=np.float32), M_total)[0] for lab in aug_labels])
            min_x, min_y = np.min(all_pts, axis=0)
            max_x, max_y = np.max(all_pts, axis=0)
            margin_x, margin_y = w_orig * 0.2, h_orig * 0.2
            
            if min_x < margin_x or max_x > w_orig - margin_x or min_y < margin_y or max_y > h_orig - margin_y:
                cx_test, cy_test = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
                bw_test, bh_test = max_x - min_x, max_y - min_y
                corr_scale = min(1.0, (w_orig - 2*margin_x) / max(bw_test, 1e-6), (h_orig - 2*margin_y) / max(bh_test, 1e-6))
                new_bw, new_bh = bw_test * corr_scale, bh_test * corr_scale
                
                v_cx_min, v_cx_max = margin_x + new_bw/2.0, w_orig - margin_x - new_bw/2.0
                v_cy_min, v_cy_max = margin_y + new_bh/2.0, h_orig - margin_y - new_bh/2.0
                new_cx = np.clip(cx_test, min(v_cx_min, v_cx_max), max(v_cx_min, v_cx_max))
                new_cy = np.clip(cy_test, min(v_cy_min, v_cy_max), max(v_cy_min, v_cy_max))
                
                T_back = np.array([[1, 0, -cx_test], [0, 1, -cy_test], [0, 0, 1]], dtype=np.float32)
                S_corr = np.array([[corr_scale, 0, 0], [0, corr_scale, 0], [0, 0, 1]], dtype=np.float32)
                T_forward = np.array([[1, 0, new_cx], [0, 1, new_cy], [0, 0, 1]], dtype=np.float32)
                M_total = (T_forward @ S_corr @ T_back) @ M_total

            # ================= 3. 执行映射与动态掩码 =================
            aug_img = cv2.warpPerspective(aug_img, M_total, (w_orig, h_orig), borderValue=(0, 0, 0))
            frame_mask = cv2.warpPerspective(np.ones((h_orig, w_orig), dtype=np.float32), M_total, (w_orig, h_orig), flags=cv2.INTER_NEAREST, borderValue=0)

            for lab in aug_labels:
                lab['pts'] = cv2.perspectiveTransform(np.array([lab['pts']], dtype=np.float32), M_total)[0]

            roi_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
            for lab in aug_labels:
                expanded_hull = get_expanded_roi(lab['pts'], self.cfg.roi_h_exp, self.cfg.roi_w_exp)
                cv2.fillPoly(roi_mask, [expanded_hull], 1.0)
            roi_mask = cv2.dilate(roi_mask, np.ones((7, 7), np.uint8), iterations=1)

        # ================= 4. 背景融合与遮挡挖洞 =================
        if random.random() < self.cfg.bg_replace_prob:
            blend_mask = roi_mask if aug_labels else np.zeros((h_orig, w_orig), dtype=np.float32)
        else:
            blend_mask = frame_mask if aug_labels else np.ones((h_orig, w_orig), dtype=np.float32)

        if aug_labels and random.random() < self.cfg.occ_prob:
            radius = min(w_orig, h_orig) * self.cfg.occ_radius_pct
            for lab in aug_labels:
                for pt in lab['pts']:
                    if random.random() < 0.5:
                        angle = random.uniform(0, 2 * np.pi)
                        cx, cy = pt[0] + random.uniform(0, radius) * np.cos(angle), pt[1] + random.uniform(0, radius) * np.sin(angle)
                        occ_w, occ_h = int(w_orig * random.uniform(*self.cfg.occ_size_pct)), int(h_orig * random.uniform(*self.cfg.occ_size_pct))
                        if random.random() < 0.5: occ_w, occ_h = occ_h, occ_w 
                        x1, y1 = int(cx - occ_w/2), int(cy - occ_h/2)
                        cv2.rectangle(blend_mask, (x1, y1), (x1 + occ_w, y1 + occ_h), 0, -1)

        blend_mask = cv2.GaussianBlur(blend_mask, (7, 7), 0)
        blend_3d = np.expand_dims(blend_mask, axis=-1)
        aug_img = (aug_img.astype(np.float32) * blend_3d + bg_img.astype(np.float32) * (1 - blend_3d)).astype(np.uint8)

        if aug_labels:
            for lab in aug_labels:
                out_count = sum(1 for pt in lab['pts'] if pt[0] < 0 or pt[0] >= w_orig or pt[1] < 0 or pt[1] >= h_orig)
                if out_count >= 3:  
                    lab['vis'] = 0

        return aug_img, aug_labels

    # =========================================================================
    # 供 train.py 在主进程中调用 (GPU 环境)
    # =========================================================================
    def process_gpu(self, batch_imgs: torch.Tensor) -> torch.Tensor:
        """
        在显存中以 Batch 为单位执行像素级运算
        输入输出均为张量: [B, C, H, W], RGB 通道, 范围 [0.0, 1.0]
        """
        device = batch_imgs.device
        
        with torch.no_grad():
            # 1. 色彩与亮度增强 (通过 torchvision 接口对全 Batch 并行计算)
            if random.random() < self.cfg.brightness_prob:
                factor = random.uniform(self.cfg.brightness_range[0], self.cfg.brightness_range[1])
                batch_imgs = TF.adjust_brightness(batch_imgs, factor)
                
            if random.random() < self.cfg.hsv_prob:
                # torchvision 的 adjust_hue 接受的 factor 在 [-0.5, 0.5] 之间，这里将配置参数映射过去
                h_factor = random.uniform(-self.cfg.hsv_h_gain, self.cfg.hsv_h_gain)
                s_factor = random.uniform(1.0 - self.cfg.hsv_s_gain, 1.0 + self.cfg.hsv_s_gain)
                
                batch_imgs = TF.adjust_hue(batch_imgs, h_factor)
                batch_imgs = TF.adjust_saturation(batch_imgs, s_factor)
                
            # 2. 高斯模糊
            if random.random() < self.cfg.blur_prob:
                ksize = random.choice(self.cfg.blur_ksize)
                # kernel_size 必须为奇数
                batch_imgs = TF.gaussian_blur(batch_imgs, kernel_size=[ksize, ksize])
                
            # 3. 高斯噪声 (显存中直接生成随机矩阵与原图相加，速度极快)
            if random.random() < self.cfg.noise_prob:
                noise = torch.randn_like(batch_imgs) * (25.0 / 255.0)
                batch_imgs = torch.clamp(batch_imgs + noise, 0.0, 1.0)
                
            # 4. 全局张量级光晕 Bloom
            if random.random() < self.cfg.bloom_prob:
                # 设置高光阈值 (由于装甲板灯条通常是白点，可以粗略设定 0.75 为发光区域界限)
                threshold = 0.75
                # 过滤出画面中的高光部分
                bright_elements = torch.clamp(batch_imgs - threshold, 0.0, 1.0) / (1.0 - threshold)
                
                # 对高光部分进行超大核模糊，模拟镜头的光晕泛射效应
                bloom_ksize = random.choice([21, 31, 41])
                bloom = TF.gaussian_blur(bright_elements, kernel_size=[bloom_ksize, bloom_ksize], sigma=[5.0, 5.0])
                
                intensity = random.uniform(0.8, 1.5)
                batch_imgs = torch.clamp(batch_imgs + bloom * intensity, 0.0, 1.0)

        return batch_imgs

# ================= 测试代码 =================
if __name__ == "__main__":
    from rich.progress import track   
    from rich.console import Console  

    console = Console()

    def parse_labels_for_test(label_path):
        parsed = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    visibility = int(parts[1]) if len(parts) > 9 else 2
                    offset = 2 if len(parts) > 9 else 1
                    pts = np.array([float(x) for x in parts[offset:offset+8]]).reshape(-1, 2)
                    parsed.append({'class_id': parts[0], 'vis': visibility, 'pts': pts})
        return parsed
        
    cfg = AugmentConfig()
    pipeline = AugmentPipeline(cfg)
    
    bg_dir = Path(cfg.bg_dir)
    bg_paths = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png")) if bg_dir.exists() else []

    # 极限测试参数
    pipeline.cfg.blur_prob = 1.0
    pipeline.cfg.hsv_prob = 1.0
    pipeline.cfg.noise_prob = 1.0
    pipeline.cfg.bloom_prob = 1.0
    pipeline.cfg.flip_prob = 1.0
    pipeline.cfg.scale_prob = 1.0
    pipeline.cfg.rotate_prob = 1.0
    pipeline.cfg.translate_prob = 1.0
    pipeline.cfg.perspective_prob = 1.0
    pipeline.cfg.bg_replace_prob = 1.0
    pipeline.cfg.occ_prob = 1.0
    pipeline.cfg.brightness_prob = 1.0
    pipeline.cfg.brightness_range = (0.8, 1.5)  

    dataset_dir = Path("./data/balance")
    train_images = list((dataset_dir / "2" / "photos").glob("*.jpg"))
    train_labels_dir = dataset_dir / "2" / "labels"
    test_samples = random.sample(train_images, min(3, len(train_images))) if train_images else []
    
    out_dir = Path("./data/test/augment")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]提取了 {len(test_samples)} 张图片准备测试流水线...[/bold cyan]")
    
    for i, img_path in enumerate(test_samples):
        img = cv2.imread(str(img_path))
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        labels = parse_labels_for_test(label_path) if label_path.exists() else []
            
        for v in track(range(5), description=f"[cyan]处理样本 {img_path.stem} ({i+1}/{len(test_samples)})[/cyan]", console=console):
            # 1. 模拟 DataLoader 端 CPU 处理
            cpu_img, aug_lbls = pipeline.process_cpu(img, labels, bg_paths)
            
            # 2. 模拟向 GPU 传递的转换
            tensor_img = torch.from_numpy(cpu_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # 3. 模拟 train.py 端 GPU 处理
            gpu_img = pipeline.process_gpu(tensor_img)
            
            # 4. 转换回 numpy 以供保存查看
            final_img = (gpu_img.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            
            for lbl in aug_lbls:
                if lbl['vis'] > 0:
                    for pt in lbl['pts']:
                        cv2.circle(final_img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                        
            cv2.imwrite(str(out_dir / f"test_{img_path.stem}_v{v}.jpg"), final_img)
            
    console.print(f"[bold green]✅ 测试完毕，输出文件已保存至 {out_dir}[/bold green]")