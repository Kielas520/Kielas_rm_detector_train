import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from rich.progress import track  # 添加这一行导入

# --- 新增：关闭 OpenCV 内部多线程与 OpenCL ---
# 这非常关键，能防止多进程读取图片时 CPU 直接飙到 100% 并吃满内存
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# ---------------------------------------------------------
# 1. 目标编码逻辑 
# ---------------------------------------------------------
def encode_target(label_data, img_w=320, img_h=320, grid_w=10, grid_h=10):
    """
    将原始标签转化为 13维的特征图网格张量
    label_data: [class_id, visibility, x1, y1, x2, y2, x3, y3, x4, y4]
    """
    kpts = np.array(label_data[2:]).reshape(4, 2)
    
    x_min, y_min = np.min(kpts, axis=0)
    x_max, y_max = np.max(kpts, axis=0)
    
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    w, h = x_max - x_min, y_max - y_min
    
    cx_norm, cy_norm = cx / img_w, cy / img_h
    w_norm, h_norm = w / img_w, h / img_h
    kpts_norm = kpts / np.array([img_w, img_h])
    
    g_x = int(np.clip(cx_norm * grid_w, 0, grid_w - 1))
    g_y = int(np.clip(cy_norm * grid_h, 0, grid_h - 1))
    
    t_x = cx_norm * grid_w - g_x
    t_y = cy_norm * grid_h - g_y
    
    kpts_grid_offset = kpts_norm * np.array([grid_w, grid_h]) - np.array([g_x, g_y])
    kpts_offset_flat = kpts_grid_offset.flatten()
    
    target_vector = np.zeros(13, dtype=np.float32)
    target_vector[0] = 1.0 
    target_vector[1:5] = [t_x, t_y, w_norm, h_norm]
    target_vector[5:13] = kpts_offset_flat
    
    class_id = int(label_data[0])
    
    return target_vector, g_x, g_y, class_id

class RMArmorDataset(Dataset):
    def __init__(self, img_dir, label_dir, input_size=(320, 320), grid_size=(10, 10), transform=None, cache_device=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.grid_size = grid_size
        self.transform = transform
        
        self.samples = [f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt')]
        
        # 缓存机制配置
        self.cache_device = cache_device
        self.use_cache = cache_device is not None
        
        if self.use_cache:
            self.imgs_cache = []
            self.targets_cache = []
            self.class_cache = []
            
            print(f"正在将数据集全量预加载至 {self.cache_device}...")
            # 将每次迭代的计算逻辑提前到初始化阶段
            for sample_name in track(self.samples, description="Caching dataset"):
                img_tensor, target_tensor, class_tensor = self._process_sample(sample_name)
                
                # 移动到指定设备（内存 'cpu' 或 显存 'cuda'）
                self.imgs_cache.append(img_tensor.to(self.cache_device))
                self.targets_cache.append(target_tensor.to(self.cache_device))
                self.class_cache.append(class_tensor.to(self.cache_device))

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
        
        # 3. 初始化空白的目标 Tensor
        target_tensor = np.zeros((13, self.grid_size[1], self.grid_size[0]), dtype=np.float32)
        class_tensor = np.zeros((1, self.grid_size[1], self.grid_size[0]), dtype=np.int64)

        # 4. 读取标签并遍历所有目标
        label_path = os.path.join(self.label_dir, f"{sample_name}.txt")
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(']')[-1].strip().split() 
            label_data = [float(x) for x in parts]

            for i in range(2, len(label_data)):
                if i % 2 == 0: 
                    label_data[i] *= scale_x
                else:          
                    label_data[i] *= scale_y

            target_vec, g_x, g_y, class_id = encode_target(
                label_data, 
                img_w=self.input_size[0], img_h=self.input_size[1], 
                grid_w=self.grid_size[0], grid_h=self.grid_size[1]
            )
            
            target_tensor[:, g_y, g_x] = target_vec
            class_tensor[0, g_y, g_x] = class_id

        # 5. 转为 Tensor 并归一化
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        
        return img_tensor, torch.from_numpy(target_tensor), torch.from_numpy(class_tensor)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 如果开启缓存，直接从显存/内存列表中读取
        if self.use_cache:
            return self.imgs_cache[idx], self.targets_cache[idx], self.class_cache[idx]
        
        # 否则回退到原本的动态加载模式
        return self._process_sample(self.samples[idx])