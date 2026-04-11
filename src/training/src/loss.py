import torch
import torch.nn as nn
from torchvision.ops import complete_box_iou_loss

class RMDetLoss(nn.Module):
    def __init__(self, lambda_conf=1.0, lambda_box=2.0, lambda_pose=1.0, grid_size=(10, 10), pos_weight=50.0):
        """
        联合损失函数模块
        lambda_conf, lambda_box, lambda_pose 分别为各项损失的权重系数
        pos_weight: 正样本置信度权重，用于缓解背景网格远多于目标网格的极度不平衡问题
        """
        super().__init__()
        self.lambda_conf = lambda_conf
        self.lambda_box = lambda_box
        self.lambda_pose = lambda_pose
        self.grid_w, self.grid_h = grid_size
        
        # 【修复】增加 pos_weight 提升正样本的惩罚权重，避免模型完全偏向预测背景
        self.bce = nn.BCEWithLogitsLoss(
            reduction='mean',
            pos_weight=torch.tensor([pos_weight])
        )
        
        # 关键点像素偏移使用 Smooth L1 损失，对抗异常点具有较好鲁棒性
        self.smooth_l1 = nn.SmoothL1Loss(reduction='mean')

    def _decode_pred_boxes(self, boxes, grid_y, grid_x):
        """解码网络预测的边界框坐标"""
        tx = torch.sigmoid(boxes[:, 0])
        ty = torch.sigmoid(boxes[:, 1])
        
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

    def forward(self, pred, target):
        """
        前向传播计算损失
        pred:   [Batch, 13, H_grid, W_grid]
        target: [Batch, 13, H_grid, W_grid]
        """
        # 确保 pos_weight 所在的设备与预测张量一致
        if self.bce.pos_weight.device != pred.device:
            self.bce.pos_weight = self.bce.pos_weight.to(pred.device)

        # -----------------------------------------
        # 1. 置信度损失 (全局所有网格均参与计算)
        # -----------------------------------------
        pred_conf = pred[:, 0, :, :]
        target_conf = target[:, 0, :, :]
        loss_conf = self.bce(pred_conf, target_conf)

        # -----------------------------------------
        # 2. 提取正样本 (制作 Mask 掩码)
        # -----------------------------------------
        pos_mask = (target_conf == 1.0)
        
        # 若当前批次输入没有任何目标，直接返回置信度损失
        if not pos_mask.any():
            return loss_conf * self.lambda_conf, {
                'loss_conf': loss_conf.item(),
                'loss_box': 0.0,
                'loss_pose': 0.0,
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

        # 【修复】使用分离的解码方法
        pred_boxes = self._decode_pred_boxes(pred_boxes_raw, grid_y, grid_x)
        target_boxes = self._decode_target_boxes(target_boxes_raw, grid_y, grid_x)
        
        loss_box = complete_box_iou_loss(pred_boxes, target_boxes, reduction='mean')

        # -----------------------------------------
        # 4. 关键点损失 (Smooth L1 Loss)
        # -----------------------------------------
        pred_pose = pred[:, 5:13, :, :].permute(0, 2, 3, 1)[pos_mask]
        target_pose = target[:, 5:13, :, :].permute(0, 2, 3, 1)[pos_mask]
        
        loss_pose = self.smooth_l1(pred_pose, target_pose)

        # -----------------------------------------
        # 5. 计算带权重的总损失
        # -----------------------------------------
        total_loss = (
            self.lambda_conf * loss_conf + 
            self.lambda_box * loss_box + 
            self.lambda_pose * loss_pose
        )

        loss_dict = {
            'loss_conf': loss_conf.item(),
            'loss_box': loss_box.item(),
            'loss_pose': loss_pose.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict