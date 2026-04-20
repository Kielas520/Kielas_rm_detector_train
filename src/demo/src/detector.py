import cv2
import torch
import numpy as np
import yaml
from pathlib import Path
from src.training.src import RMDetector, decode_tensor

class Detector:
    def __init__(self, config_path="./config.yaml"):
        """
        初始化检测器
        :param config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)['kielas_rm_demo']
        
        self.device = torch.device(self.cfg.get('device', 'cpu'))
        self.input_size = tuple(self.cfg['input_size'])  # (w, h)
        self.grid_size = tuple(self.cfg['grid_size'])
        self.conf_threshold = self.cfg['conf_threshold']
        self.nms_threshold = self.cfg['nms_iou_threshold']
        self.model_type = self.cfg['model_type'].lower()
        
        # 加载推理后端
        self._init_engine()

    def _init_engine(self):
        model_path = self.cfg['model_path']
        if self.model_type == "onnx":
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
        elif self.model_type == "pytorch":
            self.model = RMDetector().to(self.device)
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
        """执行推理"""
        if self.model_type == "onnx":
            out_np = self.session.run(None, {self.input_name: data.cpu().numpy()})[0]
            return torch.from_numpy(out_np).to(self.device)
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
            # 坐标还原
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            pts = pts.astype(np.int32)
            
            # 绘制四边形边缘
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            
            # 绘制关键点和信息
            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            label = f"ID:{cls_id} {score:.2f}"
            cv2.putText(frame, label, (pts[0][0], pts[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

    def detect(self, frame):
        """
        核心外部接口
        :param frame: OpenCV 读取的 BGR 图像 (numpy array)
        :return: 绘制后的图像
        """
        if frame is None:
            return None
            
        # 1. 前处理
        input_tensor = self._preprocess(frame)
        
        # 2. 推理
        preds = self._inference(input_tensor)
        
        # 3. 后处理 (解码/NMS)
        dets = decode_tensor(
            preds, 
            is_pred=True, 
            conf_threshold=self.conf_threshold, 
            nms_iou_threshold=self.nms_threshold, 
            grid_size=self.grid_size, 
            img_size=self.input_size
        )
        
        # 4. 绘制结果 (dets[0] 是 batch 中的第一个结果)
        if len(dets) > 0 and len(dets[0]) > 0:
            result_frame = self._draw(frame.copy(), dets[0])
            return result_frame
            
        return frame

# 使用示例
if __name__ == "__main__":
    detector = Detector(config_path="./config.yaml")
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 调用检测类
        output_img = detector.detect(frame)
        
        cv2.imshow("Detection Result", output_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()