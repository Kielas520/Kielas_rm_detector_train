import os
from os import path
import cv2
import torch
import numpy as np
import random
import hashlib  
from torch.utils.data import Dataset
from rich.console import Console  
from rich.progress import track
from pathlib import Path

console = Console()  

# =========================================================
# 系统级优化：关闭 OpenCV 内部多线程与 OpenCL
# 防止多进程读取图片时 CPU 直接飙到 100% 并吃满内存
# =========================================================
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# 注意：这里去掉了旧的 from src.training.src.augment import process_data

# ---------------------------------------------------------
# 1. 目标编码逻辑 (保持不变)
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
# 2. 数据集类 (CPU & GPU 解耦管线版)
# ---------------------------------------------------------

class RMArmorDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_id, input_size=(416, 416), strides=[8, 16, 32], 
                 scale_ranges=[[0, 64], [32, 128], [96, 9999]], transform=None, data_name='', 
                 aug_pipeline=None, bg_dir=None, shared_stage=None, processed_counter=None): 
        # ▲ 注意：上面将 augment_cfg 替换为了 aug_pipeline
        
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
        self.shared_stage = shared_stage
        self.processed_counter = processed_counter 
        
        # 接收外部实例化的数据增强管线对象
        self.aug_pipeline = aug_pipeline
        
        # 仅保存目录路径，不提前搬运数据
        self.bg_dir = bg_dir
        self.bg_paths = None
        self.current_worker_stage = -1 # 用于记录当前 Worker 的洗牌状态

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        
        # ================= 1. 读取原始图像与标签 =================
        img_path = os.path.join(self.img_dir, f"{sample_name}.jpg")
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
            
            if len(data) == 9:
                vis = 2
                pts = np.array(data[1:9], dtype=np.float32).reshape(4, 2)
            else:
                vis = int(data[1])
                pts = np.array(data[2:10], dtype=np.float32).reshape(4, 2)
                
            parsed_labels.append({'class_id': cls_id, 'vis': vis, 'pts': pts})

        # ================= 2. 半在线随机洗牌机制 & 动态读图 =================
        current_stage = self.shared_stage.value if self.shared_stage is not None else 0
        
        if self.bg_dir and (self.bg_paths is None or self.current_worker_stage != current_stage):
            bg_path_obj = Path(self.bg_dir)
            self.bg_paths = list(bg_path_obj.glob("*.jpg")) + list(bg_path_obj.glob("*.png")) if bg_path_obj.exists() else []
            self.current_worker_stage = current_stage

        md5_hash = hashlib.md5(sample_name.encode('utf-8')).hexdigest()
        base_seed = int(md5_hash, 16)
        seed = (base_seed + current_stage * 100000) % (2**32)
        
        random.seed(seed)
        np.random.seed(seed)

        # ================= 3. 呼叫 CPU 端数据增强 (仅几何运算与背景融合) =================
        if self.aug_pipeline is not None:
            # 调用管线的 CPU 专用接口
            aug_img, aug_labels = self.aug_pipeline.process_cpu(img, parsed_labels, self.bg_paths)
        else:
            aug_img, aug_labels = img, parsed_labels

        # ================= 4. 色彩空间转换与全局缩放 =================
        # 转换为 RGB 后做 resize
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
            if lbl['vis'] == 0 or lbl['class_id'] not in self.keep_classes:
                continue
                
            scaled_pts = lbl['pts'].copy()
            scaled_pts[:, 0] *= scale_x
            scaled_pts[:, 1] *= scale_y

            x_min, y_min = np.min(scaled_pts, axis=0)
            x_max, y_max = np.max(scaled_pts, axis=0)
            box_w, box_h = x_max - x_min, y_max - y_min
            max_dim = max(box_w, box_h)

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

        # 转换为 Tensor，供 DataLoader 打包成 Batch
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        target_tensors = [torch.from_numpy(t) for t in target_tensors]
        class_tensors = [torch.from_numpy(c) for c in class_tensors]
        
        # 每次 Worker 处理完一张图，跨进程计数器 +1 (用于队列状态监控)
        if self.processed_counter is not None:
            with self.processed_counter.get_lock():
                self.processed_counter.value += 1
                
        return img_tensor, target_tensors, class_tensors