import os
import cv2
import torch
import numpy as np
import random
import hashlib  # <--- 新增导入
from torch.utils.data import Dataset
from rich.console import Console  
# 顶部引入 track
from rich.progress import track

console = Console()  

# --- 关闭 OpenCV 内部多线程与 OpenCL ---
# 防止多进程读取图片时 CPU 直接飙到 100% 并吃满内存
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# ---------------------------------------------------------

from src.training.src.augment import process_data 

# ---------------------------------------------------------
# 1. 目标编码逻辑 
# ---------------------------------------------------------

def encode_multi_targets(label_data, img_w=416, img_h=416, grid_w=52, grid_h=52):
    """
    返回一个列表，包含中心网格及其相邻网格的训练目标 (Center Sampling)
    """
    kpts = np.array(label_data[2:]).reshape(4, 2)
    
    # 仅使用关键点计算中心坐标，用于决定分配给哪个网格
    x_min, y_min = np.min(kpts, axis=0)
    x_max, y_max = np.max(kpts, axis=0)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    
    cx_norm, cy_norm = cx / img_w, cy / img_h
    kpts_norm = kpts / np.array([img_w, img_h])
    
    # 获取浮点网格坐标
    grid_x_float = cx_norm * grid_w
    grid_y_float = cy_norm * grid_h
    
    g_x = int(np.clip(grid_x_float, 0, grid_w - 1))
    g_y = int(np.clip(grid_y_float, 0, grid_h - 1))
    
    # 候选网格列表：至少包含中心网格
    candidates = [(g_x, g_y)]
    
    # X方向扩散
    offset_x = grid_x_float - g_x
    if offset_x < 0.5 and g_x > 0:
        candidates.append((g_x - 1, g_y))
    elif offset_x > 0.5 and g_x < grid_w - 1:
        candidates.append((g_x + 1, g_y))
        
    # Y方向扩散
    offset_y = grid_y_float - g_y
    if offset_y < 0.5 and g_y > 0:
        candidates.append((g_x, g_y - 1))
    elif offset_y > 0.5 and g_y < grid_h - 1:
        candidates.append((g_x, g_y + 1))
        
    results = []
    class_id = int(label_data[0])
    
    for (cg_x, cg_y) in candidates:
        kpts_grid_offset = kpts_norm * np.array([grid_w, grid_h]) - np.array([cg_x, cg_y])
        kpts_offset_flat = kpts_grid_offset.flatten()
        
        target_vector = np.zeros(9, dtype=np.float32)
        target_vector[0] = 1.0  # 正样本置信度标识
        target_vector[1:9] = kpts_offset_flat
        
        results.append((target_vector, cg_x, cg_y, class_id))
        
    return results

# ---------------------------------------------------------
# 2. 数据集类 (半在线增强版)
# ---------------------------------------------------------

class RMArmorDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_id, input_size=(416, 416), strides=[8, 16, 32], 
                 scale_ranges=[[0, 64], [32, 128], [96, 9999]], transform=None, data_name='', 
                 augment_cfg=None, bg_paths=None, shared_stage=None):
        
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.strides = strides
        self.transform = transform
        self.class_id = class_id
        self.keep_classes = set(class_id)
        
        self.scale_ranges = torch.tensor(scale_ranges, dtype=torch.float32)
        self.grid_sizes = [(input_size[0] // s, input_size[1] // s) for s in strides]
        
        self.samples = [f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
        
        # 半在线洗牌时钟 (来自 train.py 的 multiprocessing.Value)
        self.shared_stage = shared_stage
        
        # 增强配置
        self.augment_cfg = augment_cfg
        
        # 只保存路径列表，利用系统底层缓存动态读取，拒绝内存爆炸！
        self.bg_paths = bg_paths
        
        if self.bg_paths:
            console.print(f"✅ {data_name} 数据集已链接 {len(self.bg_paths)} 张背景图路径。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        
        # ================= 1. 读取原始图像与标签 =================
        img_path = os.path.join(self.img_dir, f"{sample_name}.jpg")
        # 统一读取为 BGR 格式，方便进行 HSV 变换等传统色彩空间增强
        img = cv2.imread(img_path) 
        
        label_path = os.path.join(self.label_dir, f"{sample_name}.txt")
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        parsed_labels = []
        for line in lines:
            line = line.strip()
            if not line: continue

            parts = line.split(']')[-1].strip().split()
            data = [float(x) for x in parts]
            
            cls_id = int(data[0])
            
            # 兼容 9 维（无可见度）和 10 维（有可见度）的原始标签
            if len(data) == 9:
                vis = 2  # 强制补齐默认可见度 2
                pts = np.array(data[1:9], dtype=np.float32).reshape(4, 2)
            else:
                vis = int(data[1])
                pts = np.array(data[2:10], dtype=np.float32).reshape(4, 2)
                
            parsed_labels.append({
                'class_id': cls_id,
                'vis': vis,
                'pts': pts
            })

        # ================= 2. 半在线随机洗牌机制 =================
        current_stage = self.shared_stage.value if self.shared_stage is not None else 0
        
        # 使用 hashlib 生成绝对稳定的 MD5，避免多进程环境下的内置 hash() 随机盐差异
        md5_hash = hashlib.md5(sample_name.encode('utf-8')).hexdigest()
        base_seed = int(md5_hash, 16)
        
        # 强制约束在 NumPy 允许的无符号 32 位整数范围内 (0 ~ 4294967295)
        seed = (base_seed + current_stage * 100000) % (2**32)
        
        random.seed(seed)
        np.random.seed(seed)

        # ================= 3. 呼叫增强黑盒 =================
        if self.augment_cfg is not None:
            # ✅ 把传入的 self.bg_imgs 改回 self.bg_paths
            aug_img, aug_labels = process_data(img, parsed_labels, self.augment_cfg, self.bg_paths)
        else:
            aug_img, aug_labels = img, parsed_labels

        # ================= 4. 色彩空间转换与全局缩放 =================
        # 所有 OpenCV 色彩、模糊增强做完后，转回 RGB 给模型
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = aug_img.shape[:2]
        
        scale_x = self.input_size[0] / orig_w
        scale_y = self.input_size[1] / orig_h
        img_resized = cv2.resize(aug_img, self.input_size)

        # ================= 5. 分配多尺度 Tensor =================
        target_tensors = []
        class_tensors = []
        for gw, gh in self.grid_sizes:
            target_tensors.append(np.zeros((9, gh, gw), dtype=np.float32))
            class_tensors.append(np.zeros((1, gh, gw), dtype=np.int64))

        for lbl in aug_labels:
            # 关键拦截：被增强算法判定为不可见，或不在训练白名单内的类别直接丢弃
            if lbl['vis'] == 0 or lbl['class_id'] not in self.keep_classes:
                continue
                
            # 缩放坐标
            scaled_pts = lbl['pts'].copy()
            scaled_pts[:, 0] *= scale_x
            scaled_pts[:, 1] *= scale_y

            # 计算尺寸以分配特征层
            x_min, y_min = np.min(scaled_pts, axis=0)
            x_max, y_max = np.max(scaled_pts, axis=0)
            box_w, box_h = x_max - x_min, y_max - y_min
            max_dim = max(box_w, box_h)

            # 重新组装回 encode_multi_targets 期待的列表格式: [class_id, vis, x1, y1...]
            flat_label_data = [lbl['class_id'], lbl['vis']] + scaled_pts.flatten().tolist()

            for scale_idx, (gw, gh) in enumerate(self.grid_sizes):
                min_s = self.scale_ranges[scale_idx, 0]
                max_s = self.scale_ranges[scale_idx, 1]

                if not (min_s <= max_dim < max_s):
                    continue

                targets_info = encode_multi_targets(
                    flat_label_data, 
                    img_w=self.input_size[0], img_h=self.input_size[1], 
                    grid_w=gw, grid_h=gh
                )
                
                for target_vec, cg_x, cg_y, cls_id in targets_info:
                    target_tensors[scale_idx][:, cg_y, cg_x] = target_vec
                    class_tensors[scale_idx][0, cg_y, cg_x] = cls_id

        # 转换为 Tensor 返回
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        target_tensors = [torch.from_numpy(t) for t in target_tensors]
        class_tensors = [torch.from_numpy(c) for c in class_tensors]
        
        return img_tensor, target_tensors, class_tensors