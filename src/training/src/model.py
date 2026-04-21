import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def keypoint_nms(pts, scores, dist_thresh=15.0):
    """
    基于关键点最小欧氏距离的 NMS
    pts: [N, 8] (4个关键点的展平坐标) 或 [N, 4, 2]
    scores: [N] (置信度)
    dist_thresh: 判定为共享灯条/角点的最小像素距离阈值
    """
    if pts.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=pts.device)

    pts = pts.view(-1, 4, 2)
    # 按置信度降序排列
    sorted_indices = torch.argsort(scores, descending=True)
    pts = pts[sorted_indices]
    
    keep = []
    suppressed = torch.zeros(len(pts), dtype=torch.bool, device=pts.device)

    for i in range(len(pts)):
        if suppressed[i]:
            continue
        
        keep.append(sorted_indices[i])
        
        if i == len(pts) - 1:
            break
            
        # 仅与尚未被抑制的检测框进行比较
        remaining_indices = torch.arange(i + 1, len(pts), device=pts.device)
        valid_mask = ~suppressed[remaining_indices]
        valid_indices = remaining_indices[valid_mask]
        
        if len(valid_indices) == 0:
            continue

        pts_i = pts[i]               # [4, 2]
        pts_j = pts[valid_indices]   # [M, 4, 2]
        
        # 计算 pts_i 的 4 个点与 pts_j 的 4 个点之间的所有两两距离
        # 扩展维度进行广播计算:
        # pts_i -> [1, 4, 1, 2], pts_j -> [M, 1, 4, 2]
        diff = pts_i.unsqueeze(0).unsqueeze(2) - pts_j.unsqueeze(1) 
        dists = torch.norm(diff, dim=-1) # [M, 4, 4]
        
        # 提取每对装甲板之间的最小关键点距离
        min_dists = dists.view(len(valid_indices), -1).min(dim=1)[0] # [M]
        
        # 如果最小点距小于阈值，说明共享了灯条，触发抑制
        suppressed[valid_indices[min_dists < dist_thresh]] = True

    return torch.tensor(keep, dtype=torch.int64, device=pts.device)

class ConvBNReLU(nn.Module):
    """标准的卷积块，支持调整 kernel_size 和 padding"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthwiseConvBlock(nn.Module):
    """深度可分离卷积块 (Depthwise Separable Convolution) - 已增加残差连接支持"""
    def __init__(self, in_channels, out_channels, stride=1, use_res=False):
        super().__init__()
        # 只有在步长为1且输入输出通道相同时，才开启残差连接
        self.use_res = use_res and (stride == 1) and (in_channels == out_channels)
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.depthwise(x)))
        out = self.relu2(self.bn2(self.pointwise(out)))
        if self.use_res:
            return x + out
        return out

class StackedBlocks(nn.Module):
    """连续堆叠多个 Block，增加网络深度和特征提取容量"""
    def __init__(self, in_channels, out_channels, num_blocks, stride=1):
        super().__init__()
        layers = []
        # 第一层负责跨通道和下采样
        layers.append(DepthwiseConvBlock(in_channels, out_channels, stride=stride, use_res=False))
        # 后续层保持通道数和分辨率，并开启残差连接
        for _ in range(num_blocks - 1):
            layers.append(DepthwiseConvBlock(out_channels, out_channels, stride=1, use_res=True))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)

class SPPF(nn.Module):
    """空间金字塔池化 (快速版)，显著增加感受野"""
    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        c_ = in_channels // 2  
        self.cv1 = ConvBNReLU(in_channels, c_, kernel_size=1, padding=0)
        self.cv2 = ConvBNReLU(c_ * 4, out_channels, kernel_size=1, padding=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class RMHead(nn.Module):
    def __init__(self, in_channels=256, reg_max=16, num_classes=12):
        super().__init__()
        self.reg_max = reg_max
        # 解耦头：分类和回归使用独立的卷积分支
        self.cls_convs = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1),
            ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.reg_convs = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1),
            ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1)
        )
        
        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.box_pred = nn.Conv2d(in_channels, 5, kernel_size=1) # conf(1) + box(4)
        self.pose_pred = nn.Conv2d(in_channels, 8 * reg_max, kernel_size=1)
        
        self._initialize_biases()

    def _initialize_biases(self):
        # 偏置初始化：让初始预测概率接近 0.01，防止训练初期负样本 Loss 爆炸
        prior_prob = 0.01
        bias_val = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.box_pred.bias[0], bias_val) # type: ignore
        nn.init.constant_(self.cls_pred.bias, bias_val) # type: ignore

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)
        return torch.cat([self.box_pred(reg_feat), self.pose_pred(reg_feat), self.cls_pred(cls_feat)], dim=1)

class RMNeck(nn.Module):
    """标准的 FPN + PAN 结构，输出 P3, P4, P5 三个尺度"""
    def __init__(self, in_channels_list=[64, 128, 256], out_channels=256):
        super().__init__()
        c3, c4, c5 = in_channels_list
        
        # Top-down
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_f5 = ConvBNReLU(c5, out_channels, 1, padding=0)
        self.conv_f4 = ConvBNReLU(c4 + out_channels, out_channels, 1, padding=0)
        self.conv_f3 = ConvBNReLU(c3 + out_channels, out_channels, 1, padding=0)
        
        # Bottom-up
        self.down_p3 = ConvBNReLU(out_channels, out_channels, 3, stride=2, padding=1)
        self.conv_p4 = ConvBNReLU(out_channels * 2, out_channels, 1, padding=0)
        self.down_p4 = ConvBNReLU(out_channels, out_channels, 3, stride=2, padding=1)
        self.conv_p5 = ConvBNReLU(out_channels + c5, out_channels, 1, padding=0)

    def forward(self, s3, s4, s5):
        # Top-down path
        f5 = self.conv_f5(s5)
        f4 = self.conv_f4(torch.cat([s4, self.up(f5)], 1))
        f3 = self.conv_f3(torch.cat([s3, self.up(f4)], 1))
        
        # Bottom-up path
        p3 = f3
        p4 = self.conv_p4(torch.cat([f4, self.down_p3(p3)], 1))
        p5 = self.conv_p5(torch.cat([f5, self.down_p4(p4)], 1))
        return p3, p4, p5 # 对应 stride 8, 16, 32

class RMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBNReLU(3, 16, stride=2)
        self.stage2 = StackedBlocks(16, 32, num_blocks=2, stride=2)   # Stride 4
        self.stage3 = StackedBlocks(32, 64, num_blocks=3, stride=2)   # Stage 3: 步长 8 (416 -> 52x52, 64通道)
        self.stage4 = StackedBlocks(64, 128, num_blocks=3, stride=2)  # Stage 4: 步长 16 (416 -> 26x26, 128通道)
        self.stage5 = StackedBlocks(128, 256, num_blocks=3, stride=2) # Stage 5: 步长 32 (416 -> 13x13, 256通道)
        
        self.sppf = SPPF(256, 256, k=5)

    def forward(self, x):
        x = self.stage1(x)
        feat_s2 = self.stage2(x)
        feat_s3 = self.stage3(feat_s2)
        feat_s4 = self.stage4(feat_s3)
        feat_s5 = self.stage5(feat_s4)
        feat_s5 = self.sppf(feat_s5)
        
        # 仅抛出 S3, S4, S5 参与 FPN 融合
        return feat_s3, feat_s4, feat_s5

class RMDetector(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.backbone = RMBackbone()
        self.neck = RMNeck()
        self.head = RMHead(reg_max=reg_max)

    def forward(self, x):
        feats = self.backbone(x)
        ps = self.neck(*feats)
        # 对三个尺度分别预测，返回列表
        return [self.head(p) for p in ps]

def decode_tensor(tensor, is_pred=True, class_tensor=None, conf_threshold=0.5, kpt_dist_thresh = 15.0, grid_size=(52, 52), reg_max=16, img_size=(416, 416)):
    batch_size = tensor.shape[0]
    grid_w, grid_h = grid_size
    img_w, img_h = img_size
    
    if is_pred:
        conf = torch.sigmoid(tensor[:, 0, :, :])
    else:
        conf = tensor[:, 0, :, :]
        
    batch_results = []
    pose_start = 5
    pose_end = 5 + 8 * reg_max
    
    for b in range(batch_size):
        mask = conf[b] >= conf_threshold
        if not mask.any():
            batch_results.append([])
            continue
            
        grid_y, grid_x = torch.nonzero(mask, as_tuple=True)
        scores = conf[b, grid_y, grid_x]
        
        if is_pred:
            cls_logits = tensor[b, pose_end : pose_end + 12, grid_y, grid_x].T
            classes = torch.argmax(cls_logits, dim=1).float()
        else:
            if class_tensor is not None:
                classes = class_tensor[b, 0, grid_y, grid_x].float()
            else:
                classes = torch.zeros_like(scores)
        
        if is_pred:
            # 提取 DFL 分布并还原为偏移坐标
            raw_pose_dist = tensor[b, pose_start:pose_end, grid_y, grid_x].T 
            raw_pose_dist = raw_pose_dist.view(-1, 8, reg_max)
            
            prob = F.softmax(raw_pose_dist, dim=-1)
            project = torch.arange(reg_max, dtype=torch.float32, device=tensor.device)
            continuous_pose = (prob * project).sum(dim=-1)
            
            # 平移回实际偏移量
            decoded_pose_offset = continuous_pose - (reg_max // 2)
        else:
            decoded_pose_offset = tensor[b, 5:13, grid_y, grid_x].T
            
        decoded_pose = torch.zeros_like(decoded_pose_offset)
        
        for i in range(4): 
            px_offset = decoded_pose_offset[:, i*2]
            py_offset = decoded_pose_offset[:, i*2 + 1]
            
            px_norm = (px_offset + grid_x) / grid_w
            py_norm = (py_offset + grid_y) / grid_h
            
            decoded_pose[:, i*2] = px_norm * img_w
            decoded_pose[:, i*2 + 1] = py_norm * img_h
        
        # 替换为基于关键点距离的 NMS，阈值设为 15 个像素（可根据实际画面大小微调）
        
        keep_idx = keypoint_nms(decoded_pose, scores, dist_thresh=kpt_dist_thresh)
        
        scores = scores[keep_idx]
        classes = classes[keep_idx] 
        decoded_pose = decoded_pose[keep_idx]
        
        dets = torch.cat([scores.unsqueeze(1), classes.unsqueeze(1), decoded_pose], dim=1)
        batch_results.append(dets.detach().cpu().numpy())
        
    return batch_results

if __name__ == "__main__":
    model = RMBackbone()
    dummy_input = torch.randn(1, 3, 416, 416)
    out3, out4, out5 = model(dummy_input)
    print(f"Stage 3 Output Shape: {out3.shape}") 
    print(f"Stage 4 Output Shape: {out4.shape}") 
    print(f"Stage 5 Output Shape: {out5.shape}") 
    
    # RMDetector 实例化时默认 reg_max=16
    detector = RMDetector(reg_max=16)
    output = detector(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状 (多尺度列表): {[o.shape for o in output]}") 
    # 预期输出形状: [torch.Size([1, 145, 52, 52]), torch.Size([1, 145, 26, 26]), torch.Size([1, 145, 13, 13])]