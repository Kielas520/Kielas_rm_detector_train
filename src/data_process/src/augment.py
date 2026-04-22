import cv2
import random
import numpy as np
import copy
from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
@dataclass
class AugmentConfig:
    # --- 原有光学与形变参数保留 ... ---
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

    flip_prob: float = 0.5
    scale_prob: float = 0.9
    scale_range: Tuple[float, float] = (0.3, 2.5) # 兼容YAML，但在代码里严格作为“面积占比倍数”并限制安全边界
    rotate_prob: float = 0.8
    rotate_range: Tuple[float, float] = (-45, 45)
    translate_prob: float = 0.8
    translate_range: float = 0.4
    perspective_prob: float = 0.8
    perspective_factor: float = 0.35

    bg_replace_prob: float = 0.85
    bg_dir: str = "./background"
    
    # === 新增：ROI 多边形拉伸倍率 ===
    roi_h_exp: float = 2.0  # 灯条高度方向外扩倍率
    roi_w_exp: float = 1.5  # 左右灯条宽度方向外扩倍率

    occ_prob: float = 0.8
    occ_radius_pct: float = 0.4
    occ_size_pct: Tuple[float, float] = (0.02, 0.35)
    
    def __post_init__(self):
        if self.blur_ksize is None:
            self.blur_ksize = [3, 5, 7, 9, 11]

def get_expanded_roi(pts, h_exp, w_exp):
    """基于两根灯条的向量方向，向外拉伸多边形以覆盖完整装甲板"""
    # 0:左下, 1:左上, 2:右下, 3:右上
    p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
    
    # 左右灯条的中心点与方向向量
    cl = (p0 + p1) / 2.0
    vl = p1 - p0  
    cr = (p2 + p3) / 2.0
    vr = p3 - p2  

    # 宽度向量 (从左灯条中心指向右灯条中心)
    vw = cr - cl
    W = np.linalg.norm(vw)
    dw = vw / W if W > 0 else np.array([1.0, 0.0])

    # 1. 沿灯条方向拉伸高度
    p1_new = cl + (vl / 2.0) * h_exp
    p0_new = cl - (vl / 2.0) * h_exp
    p3_new = cr + (vr / 2.0) * h_exp
    p2_new = cr - (vr / 2.0) * h_exp

    # 2. 沿中心连线法向拉伸宽度
    offset = dw * W * (w_exp - 1.0) / 2.0
    p1_new -= offset
    p0_new -= offset
    p3_new += offset
    p2_new += offset

    return np.array([p0_new, p1_new, p3_new, p2_new], dtype=np.int32)

def generate_composite_bg(bg_paths, w, h):
    """生成复合堆叠背景"""
    if not bg_paths: return np.zeros((h, w, 3), dtype=np.uint8)
    bg = cv2.imread(str(random.choice(bg_paths)))
    if bg is None: return np.zeros((h, w, 3), dtype=np.uint8)
    bg = cv2.resize(bg, (w, h))

    if random.random() < 0.6:
        for _ in range(random.randint(1, 2)):
            patch = cv2.imread(str(random.choice(bg_paths)))
            if patch is None: continue
            pw, ph = random.randint(int(w*0.3), int(w*0.7)), random.randint(int(h*0.3), int(h*0.7))
            patch = cv2.resize(patch, (pw, ph))
            px, py = random.randint(0, w - pw), random.randint(0, h - ph)
            bg[py:py+ph, px:px+pw] = patch
    return bg

import cv2
import random
import numpy as np
import copy

def get_expanded_roi(pts, h_exp, w_exp):
    """基于两根灯条的向量方向，向外拉伸多边形以覆盖完整装甲板"""
    # 0:左下, 1:左上, 2:右下, 3:右上
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
    """生成复合堆叠背景"""
    if not bg_paths: return np.zeros((h, w, 3), dtype=np.uint8)
    bg = cv2.imread(str(random.choice(bg_paths)))
    if bg is None: return np.zeros((h, w, 3), dtype=np.uint8)
    bg = cv2.resize(bg, (w, h))

    if random.random() < 0.6:
        for _ in range(random.randint(1, 2)):
            patch = cv2.imread(str(random.choice(bg_paths)))
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
    
    bg_img = generate_composite_bg(bg_paths, w_orig, h_orig) if bg_paths else np.zeros_like(aug_img)

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

    # ================= 2. 核心几何变换准备 =================
    if aug_labels:
        # --- 先做绝对翻转 ---
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

        scale = 1.0
        if random.random() < cfg.scale_prob:
            target_area_ratio = random.uniform(*cfg.scale_range)
            target_area = w_orig * h_orig * target_area_ratio
            scale = np.sqrt(target_area / plate_area)
        
        tx, ty = 0.0, 0.0
        if random.random() < cfg.translate_prob:
            dt_x, dt_y = cfg.translate_range * w_orig, cfg.translate_range * h_orig
            tx = random.uniform(-dt_x, dt_x)
            ty = random.uniform(-dt_y, dt_y)
            # 初步约束中心点不飞得太离谱
            ncx, ncy = cx + tx, cy + ty
            ncx = np.clip(ncx, w_orig * 0.1, w_orig * 0.9)
            ncy = np.clip(ncy, h_orig * 0.1, h_orig * 0.9)
            tx, ty = ncx - cx, ncy - cy

        angle = random.uniform(*cfg.rotate_range) if random.random() < cfg.rotate_prob else 0.0

        # --- 组装基础 3x3 变换矩阵 M_total ---
        T1 = np.eye(3, dtype=np.float32)
        T1[0, 2], T1[1, 2] = -cx, -cy
        
        R = np.vstack([cv2.getRotationMatrix2D((0, 0), angle, scale), [0, 0, 1]]).astype(np.float32)
        
        T2 = np.eye(3, dtype=np.float32)
        T2[0, 2], T2[1, 2] = cx + tx, cy + ty
        
        M_affine = T2 @ R @ T1
        M_total = M_affine

        # 加入透视扭曲
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
            M_total = M_persp @ M_affine 

        # ================= 3. 极速安全边界校验与数学修正 =================
        # 预计算所有点经过变换后的物理坐标
        all_pts = []
        for lab in aug_labels:
            pts = cv2.perspectiveTransform(np.array([lab['pts']], dtype=np.float32), M_total)[0]
            all_pts.extend(pts)
        all_pts = np.array(all_pts)
        
        min_x, min_y = np.min(all_pts, axis=0)
        max_x, max_y = np.max(all_pts, axis=0)
        
        # 定义安全内边距 (图像边长的 0.2 倍)
        margin_x = w_orig * 0.2
        margin_y = h_orig * 0.2
        
        # 检查是否任何一个点突破了安全警戒线
        if min_x < margin_x or max_x > w_orig - margin_x or min_y < margin_y or max_y > h_orig - margin_y:
            # 触发兜底机制：计算当前点阵的包围盒状态
            cx_test = (min_x + max_x) / 2.0
            cy_test = (min_y + max_y) / 2.0
            bw_test = max_x - min_x
            bh_test = max_y - min_y
            
            # 允许的最大安全尺寸
            safe_w = w_orig - 2 * margin_x
            safe_h = h_orig - 2 * margin_y
            
            # 计算将包围盒压回安全尺寸所需的精准缩放倍率
            corr_scale = min(1.0, safe_w / max(bw_test, 1e-6), safe_h / max(bh_test, 1e-6))
            
            new_bw = bw_test * corr_scale
            new_bh = bh_test * corr_scale
            
            # 计算修补后的中心点应在哪，确保无论原矩阵平移多远，纠偏后都在安全区内
            valid_cx_min = margin_x + new_bw / 2.0
            valid_cx_max = w_orig - margin_x - new_bw / 2.0
            valid_cy_min = margin_y + new_bh / 2.0
            valid_cy_max = h_orig - margin_y - new_bh / 2.0
            
            # 消除因浮点误差导致的区间倒置
            valid_cx_min, valid_cx_max = min(valid_cx_min, valid_cx_max), max(valid_cx_min, valid_cx_max)
            valid_cy_min, valid_cy_max = min(valid_cy_min, valid_cy_max), max(valid_cy_min, valid_cy_max)
            
            new_cx = np.clip(cx_test, valid_cx_min, valid_cx_max)
            new_cy = np.clip(cy_test, valid_cy_min, valid_cy_max)
            
            # 构建修正矩阵：先移回原点进行缩放修正，再移动到合法的中心
            T_back = np.array([[1, 0, -cx_test], [0, 1, -cy_test], [0, 0, 1]], dtype=np.float32)
            S_corr = np.array([[corr_scale, 0, 0], [0, corr_scale, 0], [0, 0, 1]], dtype=np.float32)
            T_forward = np.array([[1, 0, new_cx], [0, 1, new_cy], [0, 0, 1]], dtype=np.float32)
            
            M_corr = T_forward @ S_corr @ T_back
            
            # 叠加纠偏矩阵
            M_total = M_corr @ M_total

        # ================= 4. 一次性执行所有坐标映射 =================
        aug_img = cv2.warpPerspective(aug_img, M_total, (w_orig, h_orig), borderValue=(0, 0, 0))
        frame_mask = cv2.warpPerspective(np.ones((h_orig, w_orig), dtype=np.float32), M_total, (w_orig, h_orig), flags=cv2.INTER_NEAREST, borderValue=0)

        for lab in aug_labels:
            lab['pts'] = cv2.perspectiveTransform(np.array([lab['pts']], dtype=np.float32), M_total)[0]

        # ================= 5. 后置动态生成目标蒙版 =================
        roi_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
        for lab in aug_labels:
            expanded_hull = get_expanded_roi(lab['pts'], cfg.roi_h_exp, cfg.roi_w_exp)
            cv2.fillPoly(roi_mask, [expanded_hull], 1.0)
        roi_mask = cv2.dilate(roi_mask, np.ones((7, 7), np.uint8), iterations=1)

    # ================= 6. 背景融合与遮挡 =================
    if bg_paths and random.random() < cfg.bg_replace_prob:
        blend_mask = roi_mask if aug_labels else np.zeros((h_orig, w_orig), dtype=np.float32)
    else:
        blend_mask = frame_mask if aug_labels else np.ones((h_orig, w_orig), dtype=np.float32)

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
    aug_img = (aug_img.astype(np.float32) * blend_3d + bg_img.astype(np.float32) * (1 - blend_3d)).astype(np.uint8)

    # ================= 7. 最终画质劣化与边界检查 =================
    if random.random() < cfg.blur_prob:
        ksize = random.choice(cfg.blur_ksize)
        aug_img = cv2.blur(aug_img, (ksize, ksize))

    if random.random() < cfg.noise_prob:
        noise = np.random.normal(0, 20, aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if aug_labels and random.random() < cfg.bloom_prob:
        bloom_layer = np.zeros_like(aug_img, dtype=np.float32)
        for lab in aug_labels:
            center = np.mean(lab['pts'], axis=0).astype(int)
            cv2.circle(bloom_layer, tuple(center), int(min(h_orig, w_orig) * 0.05), (255, 255, 255), -1)
        bloom_layer = cv2.GaussianBlur(bloom_layer, (31, 31), sigmaX=20)
        aug_img = np.clip(aug_img.astype(np.float32) + bloom_layer * 0.4, 0, 255).astype(np.uint8)

    if aug_labels:
        for lab in aug_labels:
            out_count = sum(1 for pt in lab['pts'] if pt[0] < 0 or pt[0] >= w_orig or pt[1] < 0 or pt[1] >= h_orig)
            if out_count >= 3:  
                lab['vis'] = 0

    return aug_img, aug_labels

    # ================= 测试代码 =================
if __name__ == "__main__":
    import yaml  # 确保导入了 yaml 模块

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
    
    # 1. 从 config.yaml 动态加载参数范围与设定
    yaml_path = Path("config.yaml")
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
                aug_data = data.get('kielas_rm_train', {}).get('dataset', {}).get('augment', {})
                if aug_data:
                    for k, v in aug_data.items():
                        if hasattr(cfg, k):
                            # 将长度为2的list转为tuple (兼容 range 参数)
                            if isinstance(v, list) and len(v) == 2 and k != 'blur_ksize':
                                v = tuple(v)
                            setattr(cfg, k, v)
                print(f"✅ 已成功从 {yaml_path} 加载变换参数。")
            except Exception as e:
                print(f"❌ 读取 config.yaml 失败: {e}，将使用默认参数。")
    else:
        print(f"⚠️ 未找到 {yaml_path}，将使用默认参数。")

    # 2. 强制将所有执行概率设为 1.0 (确保每张图都完整经历所有增强变换)
    cfg.brightness_prob = 1.0
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
    
    dataset_dir = Path("./data/balance")
    train_images = list((dataset_dir / "0" / "photos").glob("*.jpg"))
    train_labels_dir = dataset_dir / "0" / "labels"
    
    # 抽取3张图片测试
    test_samples = random.sample(train_images, min(3, len(train_images))) if train_images else []
    
    bg_dir = Path(cfg.bg_dir)
    bg_paths = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png")) if bg_dir.exists() else []
    
    out_dir = Path("./data/test/augment")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"提取了 {len(test_samples)} 张图片准备测试极限综合增强...")
    
    for i, img_path in enumerate(test_samples):
        img = cv2.imread(str(img_path))
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        
        labels = []
        if label_path.exists():
            labels = parse_labels_for_test(label_path)
            
        # 生成10个变体观察效果
        for v in range(10):
            aug_img, aug_lbls = process_data(img, labels, cfg, bg_paths)
            
            # 画上绿点用于检查几何映射对不对
            viz_img = aug_img.copy()
            for lbl in aug_lbls:
                if lbl['vis'] > 0:
                    for pt in lbl['pts']:
                        cv2.circle(viz_img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                        
            out_path = out_dir / f"test_{img_path.stem}_v{v}.jpg"
            cv2.imwrite(str(out_path), viz_img)
            
    print(f"测试完毕，输出文件已保存至 {out_dir}")