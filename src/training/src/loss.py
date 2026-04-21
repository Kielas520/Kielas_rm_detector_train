import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WingLoss(nn.Module):
    def __init__(self, omega=10.0, epsilon=2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)

    def forward(self, pred, target):
        x = torch.abs(pred - target)
        loss = torch.where(
            x < self.omega,
            self.omega * torch.log(1 + x / self.epsilon),
            x - self.C
        )
        # 你的架构中需要保持与其他 Loss 一致的 reduction='sum' 逻辑
        return loss.sum()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # 为正负样本正确分配 alpha 和 (1 - alpha)
        alpha_factor = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DFL(nn.Module):
    """Distribution Focal Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, pred_dist, target):
        """
        pred_dist: [N, 8, reg_max] 预测概率分布
        target: [N, 8] 真实连续偏移量 (已平移至 0 ~ reg_max-1)
        """
        tl = target.long()
        tr = tl + 1
        
        wl = tr.float() - target
        wr = 1.0 - wl

        loss_l = F.cross_entropy(pred_dist.transpose(1, 2), tl, reduction='none')
        loss_r = F.cross_entropy(pred_dist.transpose(1, 2), tr, reduction='none')

        return (loss_l * wl + loss_r * wr).sum()

class Integral(nn.Module):
    """根据概率分布求期望值（用于损失函数内的连续坐标还原）"""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, reg_max - 1, reg_max))

    def forward(self, x):
        # x: [N, 8, reg_max]
        prob = F.softmax(x, dim=-1)
        # 期望值计算
        continuous_val = (prob * self.project).sum(dim=-1) # type: ignore
        return continuous_val

class RMDetLoss(nn.Module):
    def __init__(self, lambda_pose=1.5, lambda_cls=1.0, alpha=0.85, gamma=2.0, reg_max=16, omega=10.0, epsilon=2.0, num_classes=12):
        super().__init__()
        # 已彻底移除 lambda_conf 和 lambda_box
        self.lambda_pose = lambda_pose 
        self.lambda_cls = lambda_cls
        self.reg_max = reg_max
        self.num_classes = num_classes
        
        self.focal_loss = FocalLoss(alpha, gamma, reduction='sum')
        self.dfl = DFL()
        self.integral = Integral(reg_max)
        self.wing_loss = WingLoss(omega=omega, epsilon=epsilon) 

    def compute_single_scale_loss(self, pred, target, target_class, global_num_pos): 
        # pred shape: [B, num_classes + 8*reg_max, H, W]
        
        # 1. 提取分类通道 (同时承担正负样本判别任务)
        pred_cls = pred[:, :self.num_classes, :, :]
        
        # target_conf 依然用于指示该网格是否有目标 (1.0 为有，0.0 为背景)
        target_conf = target[:, 0, :, :]
        pos_mask = (target_conf == 1.0)
        
        # 构建 one-hot 格式的分类 target
        target_cls_onehot = torch.zeros_like(pred_cls)
        target_class_long = target_class[:, 0:1, :, :].long()
        # 将对应类别的通道置为 1.0
        target_cls_onehot.scatter_(1, target_class_long, 1.0)
        # 将非正样本的网格全部置为 0.0 (背景)
        target_cls_onehot = target_cls_onehot * pos_mask.unsqueeze(1).float()
        
        # 计算全局分类损失 (包含正负样本)
        loss_cls_sum = self.focal_loss(pred_cls, target_cls_onehot)
        # 【关键修复】：使用全局正样本数量进行归一化
        loss_cls = loss_cls_sum / global_num_pos

        if not pos_mask.any():
            return loss_cls * self.lambda_cls, {
                'loss_pose': 0.0,
                'loss_cls': loss_cls.item(),
                'total_loss': (loss_cls * self.lambda_cls).item()
            }
        
        # 2. 关键点损失 (DFL + L1 + OKS + Structural)
        pose_start_idx = self.num_classes
        pose_end_idx = self.num_classes + 8 * self.reg_max
        
        pred_pose_raw = pred[:, pose_start_idx:pose_end_idx, :, :].permute(0, 2, 3, 1)[pos_mask]
        pred_pose_dist = pred_pose_raw.view(-1, 8, self.reg_max) 
        
        # 【注意】：假设 datasets.py 的 target_vector 已经去除了 4 维 box，精简为 9 维 (1 conf + 8 pose)
        # 因此，这里的坐标切片从 5:13 改为 1:9
        target_pose = target[:, 1:9, :, :].permute(0, 2, 3, 1)[pos_mask]
        
        target_pose_shifted = target_pose + (self.reg_max // 2)
        target_pose_shifted = torch.clamp(target_pose_shifted, 0, self.reg_max - 1.01)
        
        # 计算 DFL (除以 global_num_pos 和 8个关键点坐标)
        loss_pose_dfl_sum = self.dfl(pred_pose_dist, target_pose_shifted)
        loss_pose_dfl = loss_pose_dfl_sum / (global_num_pos * 8)
        
        pred_pose_continuous = self.integral(pred_pose_dist) - (self.reg_max // 2)
        
        # L1 Loss (改为 sum 并全局归一化) -> 替换为 Wing Loss
        loss_pose_l1_sum = self.wing_loss(pred_pose_continuous, target_pose)
        loss_pose_l1 = loss_pose_l1_sum / (global_num_pos * 8)
        
        # --- 新增：装甲板刚体几何先验约束 (Structural Loss) ---
        pred_pts = pred_pose_continuous.view(-1, 4, 2)
        target_pts = target_pose.view(-1, 4, 2)
        
        # 获取预测对角线中点 (0与2为一条对角线, 1与3为另一条)
        pred_diag1_mid = (pred_pts[:, 0, :] + pred_pts[:, 2, :]) / 2.0
        pred_diag2_mid = (pred_pts[:, 1, :] + pred_pts[:, 3, :]) / 2.0
        
        # 真实框整体几何中心
        target_center = target_pts.mean(dim=1)
        
        # 惩罚对角线中点与真实中心的距离差异，约束形状不发生严重畸变
        loss_struct_sum = F.smooth_l1_loss(pred_diag1_mid, target_center, reduction='sum') + \
                          F.smooth_l1_loss(pred_diag2_mid, target_center, reduction='sum')
        loss_struct = loss_struct_sum / (global_num_pos * 4)  # 2个中点各2个坐标维度
        # ----------------------------------------------------

        # 直接通过真实的 4 个关键点计算当前尺度下的包围框宽和高
        w_grid = target_pts[:, :, 0].max(dim=1)[0] - target_pts[:, :, 0].min(dim=1)[0]
        h_grid = target_pts[:, :, 1].max(dim=1)[0] - target_pts[:, :, 1].min(dim=1)[0]
        
        # 计算尺度因子 scale 用于 OKS
        scale = w_grid * h_grid + 1e-6
        
        dists_sq = ((pred_pose_continuous - target_pose) ** 2).view(-1, 4, 2).sum(dim=-1)
        oks = torch.exp(-dists_sq / (2 * scale.unsqueeze(1) * 0.05))
        
        # OKS Loss (改为 sum 并全局归一化)
        loss_oks_sum = (1.0 - oks).sum()
        loss_oks = loss_oks_sum / global_num_pos
        
        # 将 Structural Loss 加入总 Pose Loss（给与 0.2 的适中权重，不破坏原有梯度的平稳性）
        loss_pose = loss_pose_dfl + loss_pose_l1 + loss_oks * 0.5 + loss_struct * 0.2

        total_loss = (
            self.lambda_cls * loss_cls +
            self.lambda_pose * loss_pose
        )

        loss_dict = {
            'loss_pose': loss_pose.item(),
            'loss_cls': loss_cls.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    def forward(self, preds, targets, target_classes):
        total_loss = 0.0
        combined_loss_dict = {
            'loss_pose': 0.0, 
            'loss_cls': 0.0, 
            'total_loss': 0.0
        }
        
        # 【关键修复】：提前遍历所有尺度，计算全局的正样本数量
        global_num_pos = 0.0
        for i in range(len(targets)):
            global_num_pos += (targets[i][:, 0, :, :] == 1.0).sum().float()
        global_num_pos = torch.clamp(global_num_pos, min=1.0) # type: ignore
        
        num_scales = len(preds)
        for i in range(num_scales):
            loss_scale, dist_scale = self.compute_single_scale_loss(preds[i], targets[i], target_classes[i], global_num_pos)
            
            total_loss += loss_scale
            for k in combined_loss_dict:
                combined_loss_dict[k] += dist_scale[k]
        
        return total_loss, combined_loss_dict