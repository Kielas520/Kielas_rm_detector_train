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

# 修改 1：FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum', class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights # 新增：接收类别权重

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # 为正负样本正确分配 alpha 和 (1 - alpha)
        # 【修改点】：将 targets == 1.0 改为 targets > 0.5
        alpha_factor = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss

        # 新增核心逻辑：如果在对应通道上存在权重，施加乘性惩罚
        if self.class_weights is not None:
            # 此时 focal_loss 形状为 [B, C, H, W]，class_weights 形状为 [1, C, 1, 1]
            focal_loss = focal_loss * self.class_weights

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

# 修改 2：RMDetLoss
class RMDetLoss(nn.Module):
    # 修改 1：新增 negative_class_id 参数，默认设为 12 (根据你的 yaml 配置)
    def __init__(self, lambda_pose=1.5, lambda_cls=1.0, alpha=0.85, gamma=2.0, reg_max=16, omega=10.0, epsilon=2.0, num_classes=12, class_weights=None, negative_class_id=12):
        super().__init__()
        self.lambda_pose = lambda_pose 
        self.lambda_cls = lambda_cls
        self.reg_max = reg_max
        self.num_classes = num_classes
        self.negative_class_id = negative_class_id
        
        # 新增逻辑：处理传入的 class_weights 并传递给 FocalLoss
        if class_weights is not None:
            # 转换 class_weights 的形状以匹配预测输出 [B, C, H, W]
            cw_tensor = class_weights.view(1, -1, 1, 1)
        else:
            cw_tensor = None

        self.focal_loss = FocalLoss(alpha, gamma, reduction='sum', class_weights=cw_tensor)
        self.dfl = DFL()
        self.integral = Integral(reg_max)
        self.wing_loss = WingLoss(omega=omega, epsilon=epsilon)

    # 修改 2：分离分类归一化(global_num_pos)和位姿归一化(global_num_valid_pose)
    def compute_single_scale_loss(self, pred, target, target_class, global_num_pos, global_num_valid_pose): 
        pred_cls = pred[:, :self.num_classes, :, :]
        
        # target_conf 依然用于指示该网格是否有目标 (1.0 为有，0.0 为背景)
        target_conf = target[:, 0, :, :]
        target_class_long = target_class[:, 0:1, :, :].long()
        
        # conf_mask 依然用于分类损失，所有标注过的物体（包含类别12）都需要参与分类
        conf_mask = (target_conf == 1.0)
        
        # 修改 3：pos_mask 仅保留真正需要回归位姿的类别，剔除负样本类
        pos_mask = conf_mask & (target_class_long.squeeze(1) != self.negative_class_id)
        
        target_cls_onehot = torch.zeros_like(pred_cls)
        # 【修改点】：将 0.9 改为 1.0
        target_cls_onehot.scatter_(1, target_class_long, 1.0)
        target_cls_onehot = target_cls_onehot * conf_mask.unsqueeze(1).float()
        
        loss_cls_sum = self.focal_loss(pred_cls, target_cls_onehot)
        loss_cls = loss_cls_sum / global_num_pos

        if not pos_mask.any():
            return loss_cls * self.lambda_cls, {
                'loss_pose': 0.0,
                'loss_cls': loss_cls.item(),
                'total_loss': (loss_cls * self.lambda_cls).item()
            }
        
        pose_start_idx = self.num_classes
        pose_end_idx = self.num_classes + 8 * self.reg_max
        
        # 仅对有效目标（排除纯背景和类别12）提取预测位姿
        pred_pose_raw = pred[:, pose_start_idx:pose_end_idx, :, :].permute(0, 2, 3, 1)[pos_mask]
        pred_pose_dist = pred_pose_raw.view(-1, 8, self.reg_max) 
        
        target_pose = target[:, 1:9, :, :].permute(0, 2, 3, 1)[pos_mask]
        
        target_pose_shifted = target_pose + (self.reg_max // 2)
        target_pose_shifted = torch.clamp(target_pose_shifted, 0, self.reg_max - 1.01)
        
        # 修改 4：所有位姿损失使用 global_num_valid_pose 进行归一化
        loss_pose_dfl_sum = self.dfl(pred_pose_dist, target_pose_shifted)
        loss_pose_dfl = loss_pose_dfl_sum / (global_num_valid_pose * 8)
        
        pred_pose_continuous = self.integral(pred_pose_dist) - (self.reg_max // 2)
        
        loss_pose_l1_sum = self.wing_loss(pred_pose_continuous, target_pose)
        loss_pose_l1 = loss_pose_l1_sum / (global_num_valid_pose * 8)
        
        pred_pts = pred_pose_continuous.view(-1, 4, 2)
        target_pts = target_pose.view(-1, 4, 2)
        
        pred_diag1_mid = (pred_pts[:, 0, :] + pred_pts[:, 2, :]) / 2.0
        pred_diag2_mid = (pred_pts[:, 1, :] + pred_pts[:, 3, :]) / 2.0
        
        target_center = target_pts.mean(dim=1)
        
        loss_struct_sum = F.smooth_l1_loss(pred_diag1_mid, target_center, reduction='sum') + \
                          F.smooth_l1_loss(pred_diag2_mid, target_center, reduction='sum')
        loss_struct = loss_struct_sum / (global_num_valid_pose * 4)  
        
        w_grid = target_pts[:, :, 0].max(dim=1)[0] - target_pts[:, :, 0].min(dim=1)[0]
        h_grid = target_pts[:, :, 1].max(dim=1)[0] - target_pts[:, :, 1].min(dim=1)[0]
        
        scale = w_grid * h_grid + 1e-6
        
        dists_sq = ((pred_pose_continuous - target_pose) ** 2).view(-1, 4, 2).sum(dim=-1)
        oks = torch.exp(-dists_sq / (2 * scale.unsqueeze(1) * 0.05))
        
        loss_oks_sum = (1.0 - oks).sum()
        loss_oks = loss_oks_sum / global_num_valid_pose
        
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
        
        global_num_pos = 0.0
        global_num_valid_pose = 0.0
        
        # 提前遍历尺度：统计分类的正样本基数与姿态回归的有效正样本基数
        for i in range(len(targets)):
            conf_mask = (targets[i][:, 0, :, :] == 1.0)
            global_num_pos += conf_mask.sum().float()
            
            cls_long = target_classes[i][:, 0, :, :].long()
            valid_pose_mask = conf_mask & (cls_long.squeeze(1) != self.negative_class_id)
            global_num_valid_pose += valid_pose_mask.sum().float()
            
        global_num_pos = torch.clamp(global_num_pos, min=1.0) # type: ignore
        global_num_valid_pose = torch.clamp(global_num_valid_pose, min=1.0) # type: ignore
        
        num_scales = len(preds)
        for i in range(num_scales):
            loss_scale, dist_scale = self.compute_single_scale_loss(
                preds[i], targets[i], target_classes[i], global_num_pos, global_num_valid_pose
            )
            
            total_loss += loss_scale
            for k in combined_loss_dict:
                combined_loss_dict[k] += dist_scale[k]
        
        return total_loss, combined_loss_dict