import cv2
import torch
import numpy as np
import yaml
from pathlib import Path

# 导入精简模型及其关键点 NMS 与解码组件
from src.training.src.model import RMDetector, decode_tensor, keypoint_nms

class Detector:
    def __init__(self, config_path="./config.yaml"):
        """
        初始化检测器
        :param config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)['kielas_rm_demo']
        
        self.device = torch.device(self.cfg.get('device', 'cpu'))
        self.input_size = tuple(self.cfg['input_size'])  # (w, h)
        
        # 彻底替换历史参数
        self.strides = self.cfg.get('strides', [8, 16, 32])
        self.reg_max = self.cfg.get('reg_max', 16)
        self.num_classes = self.cfg.get('num_classes', 12)
        
        self.conf_threshold = self.cfg['conf_threshold']
        self.kpt_dist_thresh = self.cfg.get('kpt_dist_thresh', 15.0)
        self.model_type = self.cfg['model_type'].lower()
        
        self._init_engine()

    def _init_engine(self):
        model_path = self.cfg['model_path']
        if self.model_type == "onnx":
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        elif self.model_type == "pytorch":
            self.model = RMDetector(reg_max=self.reg_max, num_classes=self.num_classes).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        elif self.model_type == "torchscript":
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()

    def _preprocess(self, frame):
        """图像预处理"""
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img_tensor.unsqueeze(0).to(self.device)

    def _inference(self, data):
        """执行推理，适配多尺度输出"""
        if self.model_type == "onnx":
            out_nps = self.session.run(None, {self.input_name: data.cpu().numpy()})
            return [torch.from_numpy(out).to(self.device) for out in out_nps]
        else:
            with torch.no_grad():
                return self.model(data)

    def _draw(self, frame, dets):
        """在图上绘制结果"""
        orig_h, orig_w = frame.shape[:2]
        scale_x, scale_y = orig_w / self.input_size[0], orig_h / self.input_size[1]
        
        for det in dets:
            score, cls_id = det[0], int(det[1])
            pts = det[2:].reshape(4, 2)
            
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            pts = pts.astype(np.int32)
            
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            label = f"ID:{cls_id} {score:.2f}"
            cv2.putText(frame, label, (pts[0][0], pts[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

    def detect(self, frame):
        """核心外部接口"""
        if frame is None:
            return None
            
        input_tensor = self._preprocess(frame)
        preds = self._inference(input_tensor)
        
        # 兼容处理单张图片推理，获取多尺度特征并合并
        pred_dets_batch = []
        for i, s in enumerate(self.strides):
            current_grid = (self.input_size[0] // s, self.input_size[1] // s)
            pred_scale = decode_tensor(
                preds[i], 
                is_pred=True, 
                conf_threshold=self.conf_threshold, 
                kpt_dist_thresh=self.kpt_dist_thresh, 
                grid_size=current_grid, 
                reg_max=self.reg_max,
                img_size=self.input_size,
                num_classes=self.num_classes
            )
            
            # 取 batch=0 的结果
            if len(pred_scale[0]) > 0:
                pred_dets_batch.append(pred_scale[0])
                
        # 跨尺度 NMS
        if len(pred_dets_batch) > 0:
            merged_preds = np.concatenate(pred_dets_batch, axis=0)
            scores = torch.tensor(merged_preds[:, 0])
            pts = torch.tensor(merged_preds[:, 2:])
            
            keep = keypoint_nms(pts, scores, dist_thresh=self.kpt_dist_thresh)
            final_dets = merged_preds[keep.numpy()]
            
            return self._draw(frame.copy(), final_dets)
            
        return frame

if __name__ == "__main__":
    detector = Detector(config_path="./config.yaml")
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        output_img = detector.detect(frame)
        
        cv2.imshow("Detection Result", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()