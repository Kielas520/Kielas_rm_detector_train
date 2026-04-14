import torch
import torch.nn as nn
import torchvision

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


# RMHead 保持不变
class RMHead(nn.Module):
    # ... [保持你原来的代码] ...
    def __init__(self, in_channels=256):
        super().__init__()
        # 拆分为置信度头和关键点头
        # 原有的 1个 Conf + 4个 Box，正好是 5 个通道
        self.box_head = nn.Conv2d(in_channels, 5, kernel_size=1, stride=1)
        self.pose_head = nn.Conv2d(in_channels, 8, kernel_size=1, stride=1)
        self.cls_head = nn.Conv2d(in_channels, 12, kernel_size=1, stride=1) # 12 类

    def forward(self, x):
        box_out = self.box_head(x)   
        pose_out = self.pose_head(x)
        cls_out = self.cls_head(x)
        # 拼接后总通道数为 5 + 8 + 12 = 25
        out = torch.cat([box_out, pose_out, cls_out], dim=1)
        return out


class RMBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBNReLU(3, 16, stride=2)
        # 增加 Block 堆叠深度
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
        
        # 抛出 S2 参与底层几何特征的融合
        return feat_s2, feat_s3, feat_s4, feat_s5
    

class RMNeck(nn.Module):
    """逐级融合特征金字塔 (PANet 架构：融合 S5, S4, S3 以及高分辨率的 S2)"""
    def __init__(self, in_channels_list=[32, 64, 128, 256], out_channels=256):
        super().__init__()
        c2, c3, c4, c5 = in_channels_list
        
        # ================= 1. Top-Down (语义下发) =================
        # S5 上采样与降维 (适配 S4)
        self.up5_4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse4 = ConvBNReLU(c5 + c4, c4, kernel_size=1, padding=0)

        # S4 上采样与降维 (适配 S3)
        self.up4_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse3 = ConvBNReLU(c4 + c3, c3, kernel_size=1, padding=0)

        # ================= 2. Bottom-Up (几何上升) =================
        self.down2_3 = ConvBNReLU(c2, c2, kernel_size=3, stride=2, padding=1)

        # 最终聚合 (聚合 FPN的 S3 和 经过下采样的 S2)
        # 将融合后的 S3 升维到最终 Head 需要的通道数 (256)
        self.final_fuse = nn.Sequential(
            ConvBNReLU(c3 + c2, out_channels, kernel_size=1, padding=0),
            DepthwiseConvBlock(out_channels, out_channels, use_res=True)
        )

    def forward(self, feat_s2, feat_s3, feat_s4, feat_s5):
        # 阶段一：Top-down
        f4 = self.fuse4(torch.cat([feat_s4, self.up5_4(feat_s5)], dim=1))
        f3 = self.fuse3(torch.cat([feat_s3, self.up4_3(f4)], dim=1))

        # 阶段二：Bottom-up
        f2_down = self.down2_3(feat_s2)
        
        # 最终输出尺寸与 Stage 3 对齐 (例如 52x52)
        out_final = self.final_fuse(torch.cat([f3, f2_down], dim=1))
        return out_final


class RMDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = RMBackbone()
        # 初始化时明确传入四层的通道数
        self.neck = RMNeck(in_channels_list=[32, 64, 128, 256], out_channels=256)
        self.head = RMHead(in_channels=256)

    def forward(self, x):
        feat_s2, feat_s3, feat_s4, feat_s5 = self.backbone(x)
        fused_feat = self.neck(feat_s2, feat_s3, feat_s4, feat_s5)
        out = self.head(fused_feat)
        return out

# 新增：核心解码工具函数 (供推断与可视化使用)

def decode_tensor(tensor, is_pred=True, class_tensor=None, conf_threshold=0.5, nms_iou_threshold=0.45, grid_size=(52, 52), img_size=(416, 416)):
    batch_size = tensor.shape[0]
    grid_w, grid_h = grid_size
    img_w, img_h = img_size
    
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
        
        # --- 新增：解析 class_id ---
        if is_pred:
            # 取 13:25 通道计算预测类别
            cls_logits = tensor[b, 13:25, grid_y, grid_x].T
            classes = torch.argmax(cls_logits, dim=1).float()
        else:
            # 从外部传入的 class_tensor 获取真实类别
            if class_tensor is not None:
                classes = class_tensor[b, 0, grid_y, grid_x].float()
            else:
                classes = torch.zeros_like(scores)
        # ---------------------------
        
        raw_pose = tensor[b, 5:13, grid_y, grid_x].T  
        
        # === 新增约束：关键点输出激活 ===
        if is_pred:
            raw_pose = torch.sigmoid(raw_pose) * 6.0 - 3.0
            
        decoded_pose = torch.zeros_like(raw_pose)
        
        for i in range(4): 
            px_offset = raw_pose[:, i*2]
            py_offset = raw_pose[:, i*2 + 1]
            
            px_norm = (px_offset + grid_x) / grid_w
            py_norm = (py_offset + grid_y) / grid_h
            
            decoded_pose[:, i*2] = px_norm * img_w
            decoded_pose[:, i*2 + 1] = py_norm * img_h
            
        pts = decoded_pose.view(-1, 4, 2)
        min_xy, _ = torch.min(pts, dim=1) 
        max_xy, _ = torch.max(pts, dim=1) 
        boxes_for_nms = torch.cat([min_xy, max_xy], dim=1)
        
        keep_idx = torchvision.ops.nms(boxes_for_nms, scores, nms_iou_threshold)
        
        scores = scores[keep_idx]
        classes = classes[keep_idx] # 同步过滤类别
        decoded_pose = decoded_pose[keep_idx]
        
        # 拼合结果: [score, class_id, x1, y1, x2, y2, x3, y3, x4, y4]
        dets = torch.cat([scores.unsqueeze(1), classes.unsqueeze(1), decoded_pose], dim=1)
        batch_results.append(dets.detach().cpu().numpy())
        
    return batch_results

if __name__ == "__main__":
    model = RMBackbone()
    dummy_input = torch.randn(1, 3, 416, 416)
    out2, out3, out4, out5 = model(dummy_input)
    print(f"Stage 2 Output Shape: {out2.shape}") # 预期: [1, 32, 104, 104]
    print(f"Stage 3 Output Shape: {out3.shape}") # 预期: [1, 64, 52, 52]
    print(f"Stage 4 Output Shape: {out4.shape}") # 预期: [1, 128, 26, 26]
    print(f"Stage 5 Output Shape: {out5.shape}") # 预期: [1, 256, 13, 13]
    
    # 实例化完整模型
    detector = RMDetector()
    output = detector(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}") 
    # 预期输出形状: torch.Size([1, 25, 52, 52])