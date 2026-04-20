import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from rich.progress import track  
from rich.console import Console  

console = Console()  

# --- 新增：关闭 OpenCV 内部多线程与 OpenCL ---
# 这非常关键，能防止多进程读取图片时 CPU 直接飙到 100% 并吃满内存
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# ---------------------------------------------------------
# 1. 目标编码逻辑 
# ---------------------------------------------------------

def encode_multi_targets(label_data, img_w=416, img_h=416, grid_w=52, grid_h=52):
    """
    返回一个列表，包含中心网格及其相邻网格的训练目标 (Center Sampling)
    """
    kpts = np.array(label_data[2:]).reshape(4, 2)
    
    x_min, y_min = np.min(kpts, axis=0)
    x_max, y_max = np.max(kpts, axis=0)
    
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    w, h = x_max - x_min, y_max - y_min
    
    cx_norm, cy_norm = cx / img_w, cy / img_h
    w_norm, h_norm = w / img_w, h / img_h
    kpts_norm = kpts / np.array([img_w, img_h])
    
    # 获取浮点网格坐标
    grid_x_float = cx_norm * grid_w
    grid_y_float = cy_norm * grid_h
    
    g_x = int(np.clip(grid_x_float, 0, grid_w - 1))
    g_y = int(np.clip(grid_y_float, 0, grid_h - 1))
    
    # 候选网格列表：至少包含中心网格
    candidates = [(g_x, g_y)]
    
    # X方向扩散：如果偏移小于 0.5，说明偏左，拉上左边的网格；反之拉上右边
    offset_x = grid_x_float - g_x
    if offset_x < 0.5 and g_x > 0:
        candidates.append((g_x - 1, g_y))
    elif offset_x > 0.5 and g_x < grid_w - 1:
        candidates.append((g_x + 1, g_y))
        
    # Y方向扩散：如果偏移小于 0.5，说明偏上，拉上上面的网格；反之拉上下面
    offset_y = grid_y_float - g_y
    if offset_y < 0.5 and g_y > 0:
        candidates.append((g_x, g_y - 1))
    elif offset_y > 0.5 and g_y < grid_h - 1:
        candidates.append((g_x, g_y + 1))
        
    results = []
    class_id = int(label_data[0])
    
    # 为每一个候选网格重新计算相对偏移量
    for (cg_x, cg_y) in candidates:
        t_x = grid_x_float - cg_x
        t_y = grid_y_float - cg_y
        
        # 关键点基于当前分配网格的偏移
        kpts_grid_offset = kpts_norm * np.array([grid_w, grid_h]) - np.array([cg_x, cg_y])
        kpts_offset_flat = kpts_grid_offset.flatten()
        
        target_vector = np.zeros(13, dtype=np.float32)
        target_vector[0] = 1.0  # 正样本置信度
        target_vector[1:5] = [t_x, t_y, w_norm, h_norm]
        target_vector[5:13] = kpts_offset_flat
        
        results.append((target_vector, cg_x, cg_y, class_id))
        
    return results

class RMArmorDataset(Dataset):
    # 修改了初始化参数，将 grid_size 替换为 strides 列表以支持多尺度
    def __init__(self, img_dir, label_dir, class_id, input_size=(416, 416), strides=[8, 16, 32], transform=None, cache_device=None, force_no_cache=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.strides = strides
        self.transform = transform
        self.class_id = class_id
        
        # 预先计算出多尺度的 grid_size 列表 (宽度网格数, 高度网格数)
        self.grid_sizes = [(input_size[0] // s, input_size[1] // s) for s in strides]
        
        self.samples = [f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
        
        # 缓存机制配置
        self.cache_device = cache_device
        self.use_cache = cache_device is not None
        
        self.force_no_cache = force_no_cache

        if self.use_cache:
            self.imgs_cache = []
            self.targets_cache = []
            self.class_cache = []
            # 修改加载逻辑：只有当 cache_device 不为空且没有强制关闭时才预加载
            if self.cache_device is not None and not self.force_no_cache:
                console.print(f"正在将数据集全量预加载至 {self.cache_device}...")
                # 将每次迭代的计算逻辑提前到初始化阶段
                for sample_name in track(self.samples, description="Caching dataset"):
                    img_tensor, target_tensors, class_tensors = self._process_sample(sample_name)
                    
                    # 移动到指定设备（内存 'cpu' 或 显存 'cuda'）
                    self.imgs_cache.append(img_tensor.to(self.cache_device))
                    # 注意：多尺度下 targets 和 classes 现在是列表
                    self.targets_cache.append([t.to(self.cache_device) for t in target_tensors])
                    self.class_cache.append([c.to(self.cache_device) for c in class_tensors])
            else:
                # 如果强制关闭了缓存，把 use_cache 设为 False，确保 __getitem__ 走动态加载
                self.use_cache = False

    def _process_sample(self, sample_name):
        """将原先 __getitem__ 中的处理逻辑提取为一个独立方法"""
        # 1. 读取图像
        img_path = os.path.join(self.img_dir, f"{sample_name}.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # 2. 图像全局缩放
        scale_x = self.input_size[0] / orig_w
        scale_y = self.input_size[1] / orig_h
        img_resized = cv2.resize(img, self.input_size)
        
        # 3. 初始化多尺度的空白目标 Tensor 列表
        target_tensors = []
        class_tensors = []
        for gw, gh in self.grid_sizes:
            target_tensors.append(np.zeros((13, gh, gw), dtype=np.float32))
            class_tensors.append(np.zeros((1, gh, gw), dtype=np.int64))

        # 4. 读取标签并遍历所有目标
        label_path = os.path.join(self.label_dir, f"{sample_name}.txt")
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # 定义需要保留的类别 ID 白名单
        keep_classes = set(self.class_id)

        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(']')[-1].strip().split() 
            label_data = [float(x) for x in parts]

            # 提取类别 ID 和可见性
            class_id = int(label_data[0])
            vis = int(label_data[1])
            
            # 核心拦截逻辑：不在白名单或不可见目标直接当作背景忽略
            if class_id not in keep_classes or vis == 0:
                continue
            
            # 坐标按图像缩放比例进行放缩
            for i in range(2, len(label_data)):
                if i % 2 == 0: 
                    label_data[i] *= scale_x
                else:          
                    label_data[i] *= scale_y

            # 遍历每一个尺度，为当前目标编码标签
            for scale_idx, (gw, gh) in enumerate(self.grid_sizes):
                targets_info = encode_multi_targets(
                    label_data, 
                    img_w=self.input_size[0], img_h=self.input_size[1], 
                    grid_w=gw, grid_h=gh
                )
                
                # 填充到对应尺度的 Tensor 中
                for target_vec, cg_x, cg_y, cls_id in targets_info:
                    target_tensors[scale_idx][:, cg_y, cg_x] = target_vec
                    class_tensors[scale_idx][0, cg_y, cg_x] = cls_id

        # 5. 转为 Tensor 并归一化
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        
        # 将列表中的 numpy 数组统一转为 torch.Tensor
        target_tensors = [torch.from_numpy(t) for t in target_tensors]
        class_tensors = [torch.from_numpy(c) for c in class_tensors]
        
        return img_tensor, target_tensors, class_tensors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 如果开启缓存，直接从显存/内存列表中读取
        if self.use_cache:
            return self.imgs_cache[idx], self.targets_cache[idx], self.class_cache[idx]
        
        # 否则回退到原本的动态加载模式
        return self._process_sample(self.samples[idx])