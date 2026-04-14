import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss

# 新增 Focal Loss 类
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算基础的 BCE (不进行均值化)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # 获取预测概率 (pt)
        pt = torch.exp(-bce_loss)
        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class RMDetLoss(nn.Module):
    def __init__(self, lambda_conf=1.0, lambda_box=2.0, lambda_pose=1.5, lambda_cls = 1.0, alpha=0.85, gamma=2.0, grid_size=(10, 10)):
        """
        联合损失函数模块
        lambda_conf, lambda_box, lambda_pose 分别为各项损失的权重系数
        pos_weight: 正样本置信度权重，用于缓解背景网格远多于目标网格的极度不平衡问题
        """
        super().__init__()
        self.lambda_conf = lambda_conf
        self.lambda_box = lambda_box
        self.lambda_pose = lambda_pose 
        self.lambda_cls = lambda_cls
        self.grid_w, self.grid_h = grid_size
        
        # 将原有的 BCE 替换为 Focal Loss
        self.focal_loss = FocalLoss(alpha, gamma, reduction='mean')
        
        # 关键点像素偏移使用 Smooth L1 损失，对抗异常点具有较好鲁棒性
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def _decode_pred_boxes(self, boxes, grid_y, grid_x):
        """解码网络预测的边界框坐标"""
        # 修改这里：引入尺度拉伸，适配多网格目标分配
        tx = torch.sigmoid(boxes[:, 0]) * 2.0 - 0.5
        ty = torch.sigmoid(boxes[:, 1]) * 2.0 - 0.5
        
        # 【修正部分】: 放弃使用单一 Sigmoid 压榨宽高，
        # 借用 YOLOv5 的缩放策略，使得网络更容易输出归一化后的相对尺度。
        w = torch.clamp(torch.sigmoid(boxes[:, 2]), min=1e-6) # 限幅
        h = torch.clamp(torch.sigmoid(boxes[:, 3]), min=1e-6) # 1
         
        cx = (tx + grid_x) / self.grid_w
        cy = (ty + grid_y) / self.grid_h
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _decode_target_boxes(self, boxes, grid_y, grid_x):
        """
        解码真实标签的边界框坐标：
        由于 Target 已经是 [0, 1] 内的真实偏移量和归一化宽高，直接进行加法和除法运算，无需 Sigmoid。
        """
        tx, ty = boxes[:, 0], boxes[:, 1]
        w, h = boxes[:, 2], boxes[:, 3]
        
        cx = (tx + grid_x) / self.grid_w
        cy = (ty + grid_y) / self.grid_h
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, pred, target, target_class):
        # -----------------------------------------
        # 1. 置信度损失 (使用 Focal Loss)
        # -----------------------------------------
        pred_conf = pred[:, 0, :, :]
        target_conf = target[:, 0, :, :]
        # 调用更换后的损失函数
        loss_conf = self.focal_loss(pred_conf, target_conf)

        # -----------------------------------------
        # 以下提取正样本及后续逻辑保持原样
        # -----------------------------------------
        pos_mask = (target_conf == 1.0)
        
        if not pos_mask.any():
            return loss_conf * self.lambda_conf, {
                'loss_conf': loss_conf.item(),
                'loss_box': 0.0,
                'loss_pose': 0.0,
                'loss_cls': 0.0,
                'total_loss': (loss_conf * self.lambda_conf).item()
            }
        
        # 获取正样本网格在张量中的确切坐标位置
        indices = torch.nonzero(pos_mask)
        grid_y = indices[:, 1]  
        grid_x = indices[:, 2]  

        # -----------------------------------------
        # 3. 边界框损失 (CIoU Loss)
        # -----------------------------------------
        pred_boxes_raw = pred[:, 1:5, :, :].permute(0, 2, 3, 1)[pos_mask]
        target_boxes_raw = target[:, 1:5, :, :].permute(0, 2, 3, 1)[pos_mask]

        # 使用分离的解码方法
        pred_boxes = self._decode_pred_boxes(pred_boxes_raw, grid_y, grid_x)
        target_boxes = self._decode_target_boxes(target_boxes_raw, grid_y, grid_x)
        
        loss_box = complete_box_iou_loss(pred_boxes, target_boxes, reduction='mean')

        # -----------------------------------------
        # 4. 关键点损失 (Smooth L1 Loss + OKS)
        # -----------------------------------------
        pred_pose_raw = pred[:, 5:13, :, :].permute(0, 2, 3, 1)[pos_mask]
        target_pose = target[:, 5:13, :, :].permute(0, 2, 3, 1)[pos_mask]
        
        # 引入激活约束：约束物理偏移不超过 [-3, 3] 个网格
        pred_pose = torch.sigmoid(pred_pose_raw) * 6.0 - 3.0
        
        loss_pose_l1 = self.smooth_l1(pred_pose, target_pose)
        
        # OKS 几何结构约束
        w_norm = target[:, 3, :, :][pos_mask]
        h_norm = target[:, 4, :, :][pos_mask]
        scale = (w_norm * self.grid_w) * (h_norm * self.grid_h) + 1e-6
        
        dists_sq = ((pred_pose - target_pose) ** 2).view(-1, 4, 2).sum(dim=-1)
        oks = torch.exp(-dists_sq / (2 * scale.unsqueeze(1) * 0.05))
        loss_oks = 1.0 - oks.mean()
        
        # 混合 Pose Loss
        loss_pose = loss_pose_l1 + loss_oks * 0.5

        # -----------------------------------------
        # 5. 分类损失 (CrossEntropy Loss)
        # -----------------------------------------
        # 取出 13~24 通道作为分类的 logits，共 12 个类
        pred_cls = pred[:, 13:25, :, :].permute(0, 2, 3, 1)[pos_mask]
        
        # 提取真实的类别标签 (确保网络收到的是原本的 0~11 之间的 ID)
        target_cls = target_class[:, 0, :, :][pos_mask] 
        
        # 使用交叉熵计算分类损失
        loss_cls = nn.CrossEntropyLoss()(pred_cls, target_cls)

        # -----------------------------------------
        # 6. 计算带权重的总损失
        # -----------------------------------------
        lambda_cls = getattr(self, 'lambda_cls', 1.0)
        
        total_loss = (
            self.lambda_conf * loss_conf + 
            self.lambda_box * loss_box + 
            self.lambda_pose * loss_pose +
            lambda_cls * loss_cls
        )

        loss_dict = {
            'loss_conf': loss_conf.item(),
            'loss_box': loss_box.item(),
            'loss_pose': loss_pose.item(),
            'loss_cls': loss_cls.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict