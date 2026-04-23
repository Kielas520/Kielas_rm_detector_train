import yaml
import cv2
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.status import Status
from rich.panel import Panel
from rich import print as rprint

# 1. 导入模型组件
from src.training.src.model import RMDetector, decode_tensor, keypoint_nms

# 2. 导入海康相机
from tools.hik_camera.src.hik_camera import HikCamera

console = Console()

class InferenceEngine:
    def __init__(self, cfg):
        self.type = cfg['model_type'].lower()
        self.device = torch.device(cfg.get('device', 'cpu'))
        rprint(f"已加载到 {self.device}")
        self.model_path = Path(cfg['model_path'])
        
        # 获取核心参数，适配精简后的网络
        self.reg_max = cfg.get('reg_max', 16)
        self.num_classes = cfg.get('num_classes', 12)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {self.model_path.absolute()}")

        with Status(f"[bold cyan]正在加载 {self.type.upper()} 引擎...", console=console):
            if self.type == "onnx":
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
                self.session = ort.InferenceSession(str(self.model_path), providers=providers)
                self.input_name = self.session.get_inputs()[0].name
            elif self.type == "pytorch":
                # 完全同步重构后的初始化参数
                self.model = RMDetector(reg_max=self.reg_max, num_classes=self.num_classes).to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
            elif self.type == "torchscript":
                self.model = torch.jit.load(str(self.model_path), map_location=self.device)
                self.model.eval()

    def __call__(self, x_tensor):
        if self.type == "onnx":
            x_np = x_tensor.cpu().numpy()
            out_nps = self.session.run(None, {self.input_name: x_np})
            return [torch.from_numpy(out).to(self.device) for out in out_nps]
        else:
            with torch.no_grad():
                return self.model(x_tensor)

def process_multi_scale_preds(preds, strides, input_size, reg_max, conf_thresh, kpt_dist_thresh, num_classes):
    """处理多尺度预测列表，并执行跨尺度关键点 NMS"""
    if not isinstance(preds, list):
        preds = [preds]
        
    batch_size = preds[0].size(0)
    pred_dets_batch = [[] for _ in range(batch_size)]
    
    for i, s in enumerate(strides):
        current_grid = (input_size[0] // s, input_size[1] // s)
        pred_scale = decode_tensor(
            preds[i], 
            is_pred=True, 
            conf_threshold=conf_thresh, 
            kpt_dist_thresh=kpt_dist_thresh, 
            grid_size=current_grid, 
            reg_max=reg_max, 
            img_size=input_size,
            num_classes=num_classes # 传入正确分类数
        )
        
        for b in range(batch_size):
            if len(pred_scale[b]) > 0:
                pred_dets_batch[b].append(pred_scale[b])
                
    final_pred_dets = []
    for b in range(batch_size):
        if len(pred_dets_batch[b]) > 0:
            merged_preds = np.concatenate(pred_dets_batch[b], axis=0)
            scores = torch.tensor(merged_preds[:, 0])
            pts = torch.tensor(merged_preds[:, 2:])
            
            # 使用基于关键点距离的 NMS 过滤跨尺度重叠框
            keep = keypoint_nms(pts, scores, dist_thresh=kpt_dist_thresh)
            final_pred_dets.append(merged_preds[keep.numpy()])
        else:
            final_pred_dets.append([])
            
    return final_pred_dets

def draw_and_extract(frame, dets, orig_shape, input_size):
    orig_h, orig_w = orig_shape
    scale_x, scale_y = orig_w / input_size[0], orig_h / input_size[1]
    color_red = (0, 0, 255)
    color_blue = (255, 0, 0)
    color_black = (0, 0, 0)
    info_list = []

    for det in dets:
        # det 结构: [score, cls_id, px1, py1, px2, py2, px3, py3, px4, py4]
        score, cls_id = det[0], int(det[1])
        pts = det[2:].reshape(4, 2)
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        pts = pts.astype(np.int32)
        
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        info_list.append(f"{cls_id}, ({cx}, {cy})")
        
        # 判断颜色：ID小于6为红色，其他（大于等于6）为蓝色
        current_color = color_red if cls_id < 6 else color_blue
        
        for p in pts: cv2.circle(frame, tuple(p), 4, current_color, -1)
        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), current_color, 2)
        cv2.line(frame, tuple(pts[2]), tuple(pts[3]), current_color, 2)
        
        text = f"ID:{cls_id} | {score:.2f}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = cx + 20, cy - 20
        cv2.rectangle(frame, (tx-3, ty-th-3), (tx+tw+3, ty+bl+3), current_color, 1)
        cv2.rectangle(frame, (tx-2, ty-th-2), (tx+tw+2, ty+bl+2), color_black, -1)
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 1)
        cv2.line(frame, (cx, cy), (tx-3, ty-th//2), current_color, 1)

    return frame, info_list

def main():
    config_file = Path("./config.yaml")
    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)['kielas_rm_demo']
    
    input_sz = tuple(cfg['input_size'])
    strides = cfg.get('strides', [8, 16, 32])
    reg_max = cfg.get('reg_max', 16)
    num_classes = cfg.get('num_classes', 12)
    
    conf_t = cfg['conf_threshold']
    kpt_dist_t = cfg.get('kpt_dist_thresh', 15.0)
    device = torch.device(cfg.get('device', 'cpu'))
    
    engine = InferenceEngine(cfg)
    
    cam_type = cfg.get('camera_type', 'usb').lower()
    cam_index = cfg.get('camera_index', 0)
    
    if cam_type == "hik":
        cap = HikCamera(cam_index)
        if not cap.open():
            console.print("[bold red]无法打开海康相机[/bold red]")
            return
        cap.set_exposure(cfg.get('hik_exposure', 5000))
        current_exposure = cap.get_exposure()
    else:
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)

    console.print(Panel(
        f"[bold cyan]相机模式: {cam_type.upper()}[/bold cyan]\n"
        "[bold yellow]W / S[/bold yellow] : 调整曝光\n"
        "[bold yellow]Q[/bold yellow] : 退出程序", title="Camera System"
    ))

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                continue

            orig_shape = raw_frame.shape[:2]
            st = cv2.getTickCount()
            
            img = cv2.resize(raw_frame, input_sz)
            img_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            preds = engine(img_tensor)
            
            dets = process_multi_scale_preds(
                preds, 
                strides=strides, 
                input_size=input_sz, 
                reg_max=reg_max,
                conf_thresh=conf_t, 
                kpt_dist_thresh=kpt_dist_t,
                num_classes=num_classes
            )
            
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - st)
            
            out_frame, info = draw_and_extract(raw_frame, dets[0], orig_shape, input_sz)
            
            status_text = f"FPS: {fps:.1f} | Exposure: {current_exposure:.0f}"
            cv2.putText(out_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Detection", out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            elif key == ord('w'):
                if cam_type == "hik":
                    current_exposure += 500
                    cap.set_exposure(current_exposure)
                else:
                    current_exposure += 1
                    cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
            elif key == ord('s'):
                if cam_type == "hik":
                    current_exposure = max(100, current_exposure - 500)
                    cap.set_exposure(current_exposure)
                else:
                    current_exposure -= 1
                    cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)

    finally:
        if cam_type == "hik":
            cap.close()
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()