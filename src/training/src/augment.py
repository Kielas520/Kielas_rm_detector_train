import cv2
import random
import numpy as np
import copy
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path

@dataclass
class AugmentConfig:
    # --- 基础与光学增强 ---
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
    bloom_prob_area_range: Tuple[float, float] = (0.01, 0.13)  # 新增：光晕面积占比

    # --- 几何变换增强 ---
    flip_prob: float = 0.5
    scale_prob: float = 0.9
    scale_range: Tuple[float, float] = (0.3, 2.5) 
    rotate_prob: float = 0.8
    rotate_range: Tuple[float, float] = (-45, 45)
    translate_prob: float = 0.8
    translate_range: float = 0.4
    perspective_prob: float = 0.8
    perspective_factor: float = 0.35

    # --- 背景与遮挡 ---
    bg_replace_prob: float = 0.85
    bg_dir: str = "./background"
    
    # 完美对齐 YAML 新增的拉伸倍率
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

    # 沿灯条方向拉伸高度
    p1_new = cl + (vl / 2.0) * h_exp
    p0_new = cl - (vl / 2.0) * h_exp
    p3_new = cr + (vr / 2.0) * h_exp
    p2_new = cr - (vr / 2.0) * h_exp

    # 沿中心连线法向拉伸宽度
    offset = dw * W * (w_exp - 1.0) / 2.0
    p1_new -= offset
    p0_new -= offset
    p3_new += offset
    p2_new += offset

    return np.array([p0_new, p1_new, p3_new, p2_new], dtype=np.int32)

def generate_composite_bg(bg_paths, w, h):
    """生成复合背景，通过路径动态读取（操作系统会自动处理底层内存缓存）"""
    if not bg_paths: 
        # === 兜底调试背景生成器 ===
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        grid_size = max(w, h) // 15
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                if (x // grid_size + y // grid_size) % 2 == 0:
                    bg[y:y+grid_size, x:x+grid_size] = (100, 50, 150) # 紫色
                else:
                    bg[y:y+grid_size, x:x+grid_size] = (50, 150, 50)  # 绿色
        noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
        return cv2.add(bg, noise)

    # 正常的随机背景堆叠逻辑 (通过路径读取，避免多进程内存爆炸)
    bg_path = str(random.choice(bg_paths))
    bg = cv2.imread(bg_path)
    if bg is None: # 兜底防止坏图
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

def process_data(img, labels, cfg, bg_paths: list = None):
    aug_img = img.copy()
    aug_labels = copy.deepcopy(labels)
    h_orig, w_orig = aug_img.shape[:2]
    
    # 恢复使用 bg_paths
    bg_img = generate_composite_bg(bg_paths, w_orig, h_orig)

    # ================= 1. 基础光学增强 =================
    if random.random() < cfg.hsv_prob:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-cfg.hsv_h_gain, cfg.hsv_h_gain) * 180) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + random.uniform(-cfg.hsv_s_gain, cfg.hsv_s_gain)), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + random.uniform(-cfg.hsv_v_gain, cfg.hsv_v_gain)), 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < cfg.brightness_prob:
        factor = random.uniform(*cfg.brightness_range)
        aug_img = np.clip(aug_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # ================= 2. 核心几何变换与矩阵合并 =================
    if aug_labels:
        if random.random() < cfg.flip_prob:
            aug_img = cv2.flip(aug_img, 1)
            for lab in aug_labels:
                lab['pts'][:, 0] = w_orig - lab['pts'][:, 0]
                old_pts = lab['pts'].copy()
                lab['pts'][0], lab['pts'][1] = old_pts[2], old_pts[3] 
                lab['pts'][2], lab['pts'][3] = old_pts[0], old_pts[1] 

        primary_pts = aug_labels[0]['pts']
        plate_area = max(cv2.contourArea(get_expanded_roi(primary_pts, 1.0, 1.0)), 1.0)
        cx, cy = np.mean(primary_pts, axis=0)

        # --- 阶段 A：纯形状变换 (旋转 + 透视)，暂不缩放 ---
        angle = random.uniform(*cfg.rotate_range) if random.random() < cfg.rotate_prob else 0.0
        
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
        # 注意这里 scale 固定设为 1.0
        R_shape = np.vstack([cv2.getRotationMatrix2D((0, 0), angle, 1.0), [0, 0, 1]]).astype(np.float32)
        T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)
        
        M_shape = T2 @ R_shape @ T1

        if random.random() < cfg.perspective_prob:
            margin = min(h_orig, w_orig) * cfg.perspective_factor
            pts1 = np.float32([[0, 0], [w_orig, 0], [0, h_orig], [w_orig, h_orig]])
            pts2 = np.float32([
                [random.uniform(0, margin), random.uniform(0, margin)],
                [w_orig - random.uniform(0, margin), random.uniform(0, margin)],
                [random.uniform(0, margin), h_orig - random.uniform(0, margin)],
                [w_orig - random.uniform(0, margin), h_orig - random.uniform(0, margin)]
            ])
            M_persp = cv2.getPerspectiveTransform(pts1, pts2).astype(np.float32)
            M_shape = M_persp @ M_shape 

        # --- 阶段 B：面积补偿与目标绝对缩放 (基于画面比例) ---
        # 计算经过形状变换后，ROI 的新中心点和新面积
        pts_transformed = cv2.perspectiveTransform(np.array([primary_pts], dtype=np.float32), M_shape)[0]
        new_area = max(cv2.contourArea(get_expanded_roi(pts_transformed, 1.0, 1.0)), 1.0)
        new_cx, new_cy = np.mean(pts_transformed, axis=0)

        # 计算补偿比例
        if random.random() < cfg.scale_prob:
            # 此时 scale_range 代表占据整张图像面积的比例 (例如 0.001 到 0.1)
            target_area_ratio = random.uniform(*cfg.scale_range)
            target_area = w_orig * h_orig * target_area_ratio
            # 此公式自动抵消了透视带来的隐式放大/缩小，直达设定的绝对面积
            final_scale = np.sqrt(target_area / new_area)
        else:
            # 即使没有触发缩放增强，也要把透视改变的面积“还原”回原始板子面积
            final_scale = np.sqrt(plate_area / new_area)

        # --- 阶段 C：平移变换 ---
        tx, ty = 0.0, 0.0
        if random.random() < cfg.translate_prob:
            dt_x, dt_y = cfg.translate_range * w_orig, cfg.translate_range * h_orig
            tx = random.uniform(-dt_x, dt_x)
            ty = random.uniform(-dt_y, dt_y)
            # 必须使用形变后的新中心点 (new_cx, new_cy) 计算防出界限制
            ncx, ncy = np.clip(new_cx + tx, w_orig*0.1, w_orig*0.9), np.clip(new_cy + ty, h_orig*0.1, h_orig*0.9)
            tx, ty = ncx - new_cx, ncy - new_cy

        # --- 阶段 D：组装最终矩阵 ---
        # 以新中心点 (new_cx, new_cy) 为基准进行缩放，并加上平移
        T_scale_back = np.array([[1, 0, -new_cx], [0, 1, -new_cy], [0, 0, 1]], dtype=np.float32)
        S_mat = np.array([[final_scale, 0, 0], [0, final_scale, 0], [0, 0, 1]], dtype=np.float32)
        T_scale_fwd = np.array([[1, 0, new_cx + tx], [0, 1, new_cy + ty], [0, 0, 1]], dtype=np.float32)

        # 最终矩阵顺序：1. 形状改变 -> 2. 移回原点 -> 3. 施加精准缩放 -> 4. 移回原位并叠加随机平移
        M_total = (T_scale_fwd @ S_mat @ T_scale_back) @ M_shape

        if random.random() < cfg.perspective_prob:
            margin = min(h_orig, w_orig) * cfg.perspective_factor
            pts1 = np.float32([[0, 0], [w_orig, 0], [0, h_orig], [w_orig, h_orig]])
            pts2 = np.float32([
                [random.uniform(0, margin), random.uniform(0, margin)],
                [w_orig - random.uniform(0, margin), random.uniform(0, margin)],
                [random.uniform(0, margin), h_orig - random.uniform(0, margin)],
                [w_orig - random.uniform(0, margin), h_orig - random.uniform(0, margin)]
            ])
            M_persp = cv2.getPerspectiveTransform(pts1, pts2).astype(np.float32)
            M_total = M_persp @ M_total 

        # ================= 3. 极速安全边界校验 (兜底0.2边长) =================
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
            v_cx_min, v_cx_max = min(v_cx_min, v_cx_max), max(v_cx_min, v_cx_max)
            v_cy_min, v_cy_max = min(v_cy_min, v_cy_max), max(v_cy_min, v_cy_max)
            
            new_cx = np.clip(cx_test, v_cx_min, v_cx_max)
            new_cy = np.clip(cy_test, v_cy_min, v_cy_max)
            
            T_back = np.array([[1, 0, -cx_test], [0, 1, -cy_test], [0, 0, 1]], dtype=np.float32)
            S_corr = np.array([[corr_scale, 0, 0], [0, corr_scale, 0], [0, 0, 1]], dtype=np.float32)
            T_forward = np.array([[1, 0, new_cx], [0, 1, new_cy], [0, 0, 1]], dtype=np.float32)
            
            M_total = (T_forward @ S_corr @ T_back) @ M_total

        # ================= 4. 执行映射与动态掩码 =================
        aug_img = cv2.warpPerspective(aug_img, M_total, (w_orig, h_orig), borderValue=(0, 0, 0))
        frame_mask = cv2.warpPerspective(np.ones((h_orig, w_orig), dtype=np.float32), M_total, (w_orig, h_orig), flags=cv2.INTER_NEAREST, borderValue=0)

        for lab in aug_labels:
            lab['pts'] = cv2.perspectiveTransform(np.array([lab['pts']], dtype=np.float32), M_total)[0]

        roi_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
        for lab in aug_labels:
            expanded_hull = get_expanded_roi(lab['pts'], cfg.roi_h_exp, cfg.roi_w_exp)
            cv2.fillPoly(roi_mask, [expanded_hull], 1.0)
        roi_mask = cv2.dilate(roi_mask, np.ones((7, 7), np.uint8), iterations=1)

    # ================= 5. 背景融合与遮挡挖洞 =================
    if random.random() < cfg.bg_replace_prob:
        blend_mask = roi_mask if aug_labels else np.zeros((h_orig, w_orig), dtype=np.float32)
    else:
        blend_mask = frame_mask if aug_labels else np.ones((h_orig, w_orig), dtype=np.float32)

    # 在掩码上挖洞 (设为 0)，渲染时直接透出底层的背景 (bg_img)
    if aug_labels and random.random() < cfg.occ_prob:
        radius = min(w_orig, h_orig) * cfg.occ_radius_pct
        for lab in aug_labels:
            for pt in lab['pts']:
                if random.random() < 0.5:
                    angle = random.uniform(0, 2 * np.pi)
                    cx, cy = pt[0] + random.uniform(0, radius) * np.cos(angle), pt[1] + random.uniform(0, radius) * np.sin(angle)
                    occ_w, occ_h = int(w_orig * random.uniform(*cfg.occ_size_pct)), int(h_orig * random.uniform(*cfg.occ_size_pct))
                    if random.random() < 0.5: occ_w, occ_h = occ_h, occ_w 
                    
                    x1, y1 = int(cx - occ_w/2), int(cy - occ_h/2)
                    cv2.rectangle(blend_mask, (x1, y1), (x1 + occ_w, y1 + occ_h), 0, -1)

    blend_mask = cv2.GaussianBlur(blend_mask, (7, 7), 0)
    blend_3d = np.expand_dims(blend_mask, axis=-1)
    # 此处合成：1的位置保留装甲板，0的位置（原图死黑边、遮挡孔洞、全屏被替换区）透出背景图
    aug_img = (aug_img.astype(np.float32) * blend_3d + bg_img.astype(np.float32) * (1 - blend_3d)).astype(np.uint8)

    # ================= 6. 最终画质劣化 (应用在混合后的图像上) =================
    if random.random() < cfg.blur_prob:
        ksize = random.choice(cfg.blur_ksize)
        aug_img = cv2.blur(aug_img, (ksize, ksize))

    if random.random() < cfg.noise_prob:
        noise = np.random.normal(0, 25, aug_img.shape).astype(np.float32) # 加强了噪声体现
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if aug_labels and random.random() < cfg.bloom_prob:
        for lab in aug_labels:
            pts = lab['pts']
            
            # 1. 获取 BBox 边界并在内部随机游走生成光晕中心
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            
            cx = int(random.uniform(min_x, max_x))
            cy = int(random.uniform(min_y, max_y))

            # 2. 根据装甲板面积比例动态计算【高亮核心半径】
            plate_area = max(1.0, cv2.contourArea(pts.astype(np.float32)))
            area_ratio = random.uniform(*cfg.bloom_prob_area_range)
            bloom_area = plate_area * area_ratio
            
            # 圆面积公式反推高亮核心的半径
            core_radius = int(np.sqrt(bloom_area / np.pi))

            # 如果算出来的核心太小（比如远距离的小板子），跳过光晕避免满屏噪点
            if core_radius < 2:
                continue

            # 3. 核心修复：生成局部高斯光晕贴图，贴图必须比核心大以容纳自然衰减
            patch_radius = core_radius * 3  # 贴图范围是核心的 3 倍
            size = patch_radius * 2 + 1
            
            # 将网格映射到 -3 到 3 的空间
            x_grid, y_grid = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
            d = np.sqrt(x_grid**2 + y_grid**2)

            # 控制衰减速度：当 sigma=1.0 时，d=1 (即恰好处于 core_radius 边界) 处亮度约保留 60%
            sigma = random.uniform(0.8, 1.2)
            g = np.exp(-(d**2 / (2.0 * sigma**2)))
            g = np.clip(g, 0, 1)

            color = np.array([random.randint(220, 255), random.randint(220, 255), random.randint(220, 255)], dtype=np.float32)
            intensity = random.uniform(0.8, 1.5)

            # 坐标边界计算必须使用扩充后的 patch_radius
            x1, y1 = cx - patch_radius, cy - patch_radius
            x2, y2 = cx + patch_radius + 1, cy + patch_radius + 1

            ix1, iy1 = max(0, x1), max(0, y1)
            ix2, iy2 = min(w_orig, x2), min(h_orig, y2)

            if ix1 < ix2 and iy1 < iy2:
                patch_g = g[iy1-y1:iy2-y1, ix1-x1:ix2-x1]
                patch_g = np.expand_dims(patch_g, axis=-1)
                
                roi = aug_img[iy1:iy2, ix1:ix2].astype(np.float32)
                roi = np.clip(roi + patch_g * color * intensity, 0, 255)
                aug_img[iy1:iy2, ix1:ix2] = roi.astype(np.uint8)

    if aug_labels:
        for lab in aug_labels:
            out_count = sum(1 for pt in lab['pts'] if pt[0] < 0 or pt[0] >= w_orig or pt[1] < 0 or pt[1] >= h_orig)
            if out_count >= 3:  
                lab['vis'] = 0

    return aug_img, aug_labels

# ================= 测试代码 =================
if __name__ == "__main__":
    import yaml

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
    
    yaml_path = Path("config.yaml")
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
                aug_data = data.get('kielas_rm_train', {}).get('dataset', {}).get('augment', {})
                if aug_data:
                    for k, v in aug_data.items():
                        if hasattr(cfg, k):
                            if isinstance(v, list) and len(v) == 2 and k != 'blur_ksize':
                                v = tuple(v)
                            setattr(cfg, k, v)
                print(f"✅ 已成功从 {yaml_path} 加载参数。")
            except Exception as e:
                print(f"❌ 读取 config.yaml 失败: {e}")
    
    bg_dir = Path(cfg.bg_dir)
    bg_paths = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png")) if bg_dir.exists() else []
    
    # 删除了测试环境的提前读入逻辑，恢复传背景图路径列表
    if not bg_paths:
        print("⚠️  警告：未找到背景图片！已自动启用紫绿网格作为测试替代背景。")

    # 1. 强制所有变换发生
    cfg.blur_prob = 1.0
    cfg.hsv_prob = 1.0
    cfg.noise_prob = 1.0
    cfg.bloom_prob = 1.0
    cfg.flip_prob = 1.0
    cfg.scale_prob = 1.0
    cfg.rotate_prob = 1.0
    cfg.translate_prob = 1.0
    cfg.perspective_prob = 1.0
    cfg.bg_replace_prob = 1.0
    cfg.occ_prob = 1.0
    
    # 2. 【关键】限制测试时的亮度，避免画面死黑导致完全看不出效果
    cfg.brightness_prob = 1.0
    cfg.brightness_range = (0.8, 1.5)  

    dataset_dir = Path("./data/balance")
    train_images = list((dataset_dir / "2" / "photos").glob("*.jpg"))
    train_labels_dir = dataset_dir / "2" / "labels"
    
    test_samples = random.sample(train_images, min(3, len(train_images))) if train_images else []
    
    out_dir = Path("./data/test/augment")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"提取了 {len(test_samples)} 张图片准备测试极限综合增强...")
    
    for i, img_path in enumerate(test_samples):
        img = cv2.imread(str(img_path))
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        
        labels = []
        if label_path.exists():
            labels = parse_labels_for_test(label_path)
            
        for v in range(30):
            # ✅ 这里恢复传入 bg_paths
            aug_img, aug_lbls = process_data(img, labels, cfg, bg_paths)
            viz_img = aug_img.copy()
            for lbl in aug_lbls:
                if lbl['vis'] > 0:
                    for pt in lbl['pts']:
                        cv2.circle(viz_img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                        
            out_path = out_dir / f"test_{img_path.stem}_v{v}.jpg"
            cv2.imwrite(str(out_path), viz_img)
            
    print(f"测试完毕，输出文件已保存至 {out_dir}")