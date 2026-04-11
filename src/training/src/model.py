import torch
import torch.nn as nn
import torchvision

class ConvBNReLU(nn.Module):
    """标准的 3x3 卷积块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthwiseConvBlock(nn.Module):
    """深度可分离卷积块 (Depthwise Separable Convolution)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))
        x = self.relu2(self.bn2(self.pointwise(x)))
        return x

# RMHead 保持不变
class RMHead(nn.Module):
    # ... [保持你原来的代码] ...
    def __init__(self, in_channels=256):
        super().__init__()
        self.box_head = nn.Conv2d(in_channels, 5, kernel_size=1, stride=1)
        self.pose_head = nn.Conv2d(in_channels, 8, kernel_size=1, stride=1)

    def forward(self, x):
        box_out = self.box_head(x)   
        pose_out = self.pose_head(x) 
        out = torch.cat([box_out, pose_out], dim=1)
        return out

class RMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBNReLU(3, 16, stride=2)
        self.stage2 = DepthwiseConvBlock(16, 32, stride=2)
        self.stage3 = DepthwiseConvBlock(32, 64, stride=2)
        self.stage4 = DepthwiseConvBlock(64, 128, stride=2)
        self.stage5 = DepthwiseConvBlock(128, 256, stride=2)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        # Stage 3: 步长 8 (416 -> 52x52, 64通道)
        feat_stage3 = self.stage3(x)
        # Stage 4: 步长 16 (416 -> 26x26, 128通道)
        feat_stage4 = self.stage4(feat_stage3)
        # Stage 5: 步长 32 (416 -> 13x13, 256通道)
        feat_stage5 = self.stage5(feat_stage4)
        
        return feat_stage3, feat_stage4, feat_stage5
    
class RMNeck(nn.Module):
    """逐级融合特征金字塔 (5融4, 4融3)"""
    def __init__(self, in_channels_s3=64, in_channels_s4=128, in_channels_s5=256, out_channels=256):
        super().__init__()
        
        # 1. S5 上采样与降维 (适配 S4)
        self.upsample_s5 = nn.Sequential(
            nn.Conv2d(in_channels_s5, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.fuse_s4 = nn.Sequential(
            ConvBNReLU(in_channels_s4 + 128, 128), 
            DepthwiseConvBlock(128, 128)
        )

        # 2. S4 上采样与降维 (适配 S3)
        self.upsample_s4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.fuse_s3 = nn.Sequential(
            # 将融合后的 S3 升维到最终 Head 需要的通道数 (256)
            ConvBNReLU(in_channels_s3 + 64, out_channels),
            DepthwiseConvBlock(out_channels, out_channels)
        )

    def forward(self, feat_s3, feat_s4, feat_s5):
        # 阶段一：S5 融 S4
        feat_s5_up = self.upsample_s5(feat_s5)
        out_s4 = torch.cat([feat_s4, feat_s5_up], dim=1) 
        out_s4 = self.fuse_s4(out_s4) 

        # 阶段二：融合后的 S4 融 S3
        out_s4_up = self.upsample_s4(out_s4)
        out_final = torch.cat([feat_s3, out_s4_up], dim=1)
        out_final = self.fuse_s3(out_final)
        
        # 最终输出尺寸与 Stage 3 对齐 (例如 52x52)
        return out_final

class RMDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = RMBackbone()
        # 初始化时明确传入三层的通道数
        self.neck = RMNeck(in_channels_s3=64, in_channels_s4=128, in_channels_s5=256, out_channels=256)
        self.head = RMHead(in_channels=256)

    def forward(self, x):
        feat_s3, feat_s4, feat_s5 = self.backbone(x)
        fused_feat = self.neck(feat_s3, feat_s4, feat_s5)
        out = self.head(fused_feat)
        return out

# 新增：核心解码工具函数 (供推断与可视化使用)

def decode_tensor(tensor, is_pred=True, conf_threshold=0.5, nms_iou_threshold=0.45, grid_size=(52, 52), img_size=(416, 416)):
    """
    将网格张量解码为像素坐标，并执行全局 NMS 后处理
    """
    batch_size = tensor.shape[0]
    grid_w, grid_h = grid_size
    img_w, img_h = img_size
    
    # 1. 只有预测值需要 Sigmoid，真实标签已经是 0 或 1
    if is_pred:
        conf = torch.sigmoid(tensor[:, 0, :, :])
    else:
        conf = tensor[:, 0, :, :]
        
    batch_results = []
    
    for b in range(batch_size):
        mask = conf[b] >= conf_threshold
        if not mask.any():
            batch_results.append([])
            continue
            
        grid_y, grid_x = torch.nonzero(mask, as_tuple=True)
        scores = conf[b, grid_y, grid_x]
        
        raw_pose = tensor[b, 5:13, grid_y, grid_x].T  
        decoded_pose = torch.zeros_like(raw_pose)
        
        for i in range(4): 
            px_offset = raw_pose[:, i*2]
            py_offset = raw_pose[:, i*2 + 1]
            
            px_norm = (px_offset + grid_x) / grid_w
            py_norm = (py_offset + grid_y) / grid_h
            
            decoded_pose[:, i*2] = px_norm * img_w
            decoded_pose[:, i*2 + 1] = py_norm * img_h
            
        # ==========================================
        # 修复：把 NMS 提出来，让 Pred 和 GT 都执行 NMS 过滤
        # ==========================================
        
        # 计算 4 个关键点的最小外接矩形 [N, 4] -> (x1, y1, x2, y2)
        pts = decoded_pose.view(-1, 4, 2)
        min_xy, _ = torch.min(pts, dim=1) 
        max_xy, _ = torch.max(pts, dim=1) 
        boxes_for_nms = torch.cat([min_xy, max_xy], dim=1)
        
        # 执行 NMS
        keep_idx = torchvision.ops.nms(boxes_for_nms, scores, nms_iou_threshold)
        
        # 根据 NMS 结果过滤
        scores = scores[keep_idx]
        decoded_pose = decoded_pose[keep_idx]
        
        # 拼合结果
        dets = torch.cat([scores.unsqueeze(1), decoded_pose], dim=1)
        batch_results.append(dets.detach().cpu().numpy())
        
    return batch_results

if __name__ == "__main__":
    model = RMBackbone()
    dummy_input = torch.randn(1, 3, 640, 640)
    out4, out5 = model(dummy_input)
    print(f"Stage 4 Output Shape: {out4.shape}") # 预期: [1, 128, 40, 40]
    print(f"Stage 5 Output Shape: {out5.shape}") # 预期: [1, 256, 20, 20]
    
    # 实例化完整模型
    detector = RMDetector()
    output = detector(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}") 
    # 预期输出形状: torch.Size([1, 13, 40, 40])
    