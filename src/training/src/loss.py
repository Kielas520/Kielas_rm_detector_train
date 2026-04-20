import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss

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

        return (loss_l * wl + loss_r * wr).mean()

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
        continuous_val = (prob * self.project).sum(dim=-1)
        return continuous_val

class RMDetLoss(nn.Module):
    def __init__(self, lambda_conf=1.0, lambda_box=2.0, lambda_pose=1.5, lambda_cls=1.0, alpha=0.85, gamma=2.0, reg_max=16):
        super().__init__()
        self.lambda_conf = lambda_conf
        self.lambda_box = lambda_box
        self.lambda_pose = lambda_pose 
        self.lambda_cls = lambda_cls
        self.reg_max = reg_max
        
        self.focal_loss = FocalLoss(alpha, gamma, reduction='sum')
        self.dfl = DFL()
        self.integral = Integral(reg_max)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def _decode_pred_boxes(self, boxes, grid_y, grid_x, grid_w, grid_h):
        tx = torch.sigmoid(boxes[:, 0]) * 2.0 - 0.5
        ty = torch.sigmoid(boxes[:, 1]) * 2.0 - 0.5
        w = torch.clamp(torch.sigmoid(boxes[:, 2]), min=1e-6)
        h = torch.clamp(torch.sigmoid(boxes[:, 3]), min=1e-6)
         
        cx = (tx + grid_x) / grid_w
        cy = (ty + grid_y) / grid_h
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _decode_target_boxes(self, boxes, grid_y, grid_x, grid_w, grid_h):
        tx, ty = boxes[:, 0], boxes[:, 1]
        w, h = boxes[:, 2], boxes[:, 3]
        
        cx = (tx + grid_x) / grid_w
        cy = (ty + grid_y) / grid_h
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def compute_single_scale_loss(self, pred, target, target_class, global_num_pos): # 新增参数 global_num_pos
        grid_h, grid_w = pred.shape[2], pred.shape[3]
        
        pred_conf = pred[:, 0, :, :]
        target_conf = target[:, 0, :, :]
        
        loss_conf_sum = self.focal_loss(pred_conf, target_conf)
        pos_mask = (target_conf == 1.0)
        
        # 【关键修复】：使用全局正样本数量进行归一化
        loss_conf = loss_conf_sum / global_num_pos

        if not pos_mask.any():
            return loss_conf * self.lambda_conf, {
                'loss_conf': loss_conf.item(),
                'loss_box': 0.0,
                'loss_pose': 0.0,
                'loss_cls': 0.0,
                'total_loss': (loss_conf * self.lambda_conf).item()
            }
        
        indices = torch.nonzero(pos_mask)
        grid_y = indices[:, 1]  
        grid_x = indices[:, 2]  

        # 2. 边界框损失
        pred_boxes_raw = pred[:, 1:5, :, :].permute(0, 2, 3, 1)[pos_mask]
        target_boxes_raw = target[:, 1:5, :, :].permute(0, 2, 3, 1)[pos_mask]

        pred_boxes = self._decode_pred_boxes(pred_boxes_raw, grid_y, grid_x, grid_w, grid_h)
        target_boxes = self._decode_target_boxes(target_boxes_raw, grid_y, grid_x, grid_w, grid_h)
        
        loss_box = complete_box_iou_loss(pred_boxes, target_boxes, reduction='mean')

        # 3. 关键点损失 (DFL + L1 + OKS)
        pose_start_idx = 5
        pose_end_idx = 5 + 8 * self.reg_max
        
        pred_pose_raw = pred[:, pose_start_idx:pose_end_idx, :, :].permute(0, 2, 3, 1)[pos_mask]
        pred_pose_dist = pred_pose_raw.view(-1, 8, self.reg_max) 
        
        target_pose = target[:, 5:13, :, :].permute(0, 2, 3, 1)[pos_mask]
        
        # 将真实坐标平移，使得 0 对应负的最大偏移
        target_pose_shifted = target_pose + (self.reg_max // 2)
        target_pose_shifted = torch.clamp(target_pose_shifted, 0, self.reg_max - 1.01)
        
        # 计算 DFL
        loss_pose_dfl = self.dfl(pred_pose_dist, target_pose_shifted)
        
        # 将预测的分布还原为连续值以计算 OKS 和 L1
        pred_pose_continuous = self.integral(pred_pose_dist) - (self.reg_max // 2)
        
        loss_pose_l1 = self.smooth_l1(pred_pose_continuous, target_pose)
        
        w_norm = target[:, 3, :, :][pos_mask]
        h_norm = target[:, 4, :, :][pos_mask]
        scale = (w_norm * grid_w) * (h_norm * grid_h) + 1e-6
        
        dists_sq = ((pred_pose_continuous - target_pose) ** 2).view(-1, 4, 2).sum(dim=-1)
        oks = torch.exp(-dists_sq / (2 * scale.unsqueeze(1) * 0.05))
        loss_oks = 1.0 - oks.mean()
        
        loss_pose = loss_pose_dfl + loss_pose_l1 + loss_oks * 0.5

        # 4. 分类损失
        pred_cls = pred[:, pose_end_idx : pose_end_idx + 12, :, :].permute(0, 2, 3, 1)[pos_mask]
        target_cls = target_class[:, 0, :, :][pos_mask] 
        loss_cls = nn.CrossEntropyLoss()(pred_cls, target_cls)

        total_loss = (
            self.lambda_conf * loss_conf + 
            self.lambda_box * loss_box + 
            self.lambda_pose * loss_pose +
            self.lambda_cls * loss_cls
        )

        loss_dict = {
            'loss_conf': loss_conf.item(),
            'loss_box': loss_box.item(),
            'loss_pose': loss_pose.item(),
            'loss_cls': loss_cls.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict

    def forward(self, preds, targets, target_classes):
        total_loss = 0.0
        combined_loss_dict = {
            'loss_conf': 0.0, 
            'loss_box': 0.0, 
            'loss_pose': 0.0, 
            'loss_cls': 0.0, 
            'total_loss': 0.0
        }
        
        # 【关键修复】：提前遍历所有尺度，计算全局的正样本数量
        global_num_pos = 0.0
        for i in range(len(targets)):
            global_num_pos += (targets[i][:, 0, :, :] == 1.0).sum().float()
        global_num_pos = torch.clamp(global_num_pos, min=1.0)
        
        num_scales = len(preds)
        for i in range(num_scales):
            loss_scale, dist_scale = self.compute_single_scale_loss(preds[i], targets[i], target_classes[i], global_num_pos)
            
            total_loss += loss_scale
            for k in combined_loss_dict:
                combined_loss_dict[k] += dist_scale[k]
        
        return total_loss / num_scales, combined_loss_dict