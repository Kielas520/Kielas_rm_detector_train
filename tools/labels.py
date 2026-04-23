import cv2
import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    """读取配置文件"""
    p = Path(config_path)
    if not p.exists():
        print(f"错误: 找不到配置文件 {config_path}")
        return None
    with open(p, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        return config.get('kielas_rm_train', {}).get('negative_sampler', {})

def get_next_index(export_dir: Path, class_id: int) -> int:
    """扫描现有文件夹，获取该类别的下一个文件序号"""
    label_dir = export_dir / str(class_id) / "labels"
    if not label_dir.exists():
        return 0
    
    # 查找最大的数字文件名
    max_idx = -1
    for f in label_dir.glob("*.txt"):
        try:
            idx = int(f.stem)
            if idx > max_idx:
                max_idx = idx
        except ValueError:
            continue
    return max_idx + 1

def save_annotation(export_dir: Path, frame, class_id: int, points: list, img_idx: int):
    """保存图片和标签"""
    base_dir = export_dir / str(class_id)
    photo_dir = base_dir / "photos"
    label_dir = base_dir / "labels"
    
    photo_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存格式: class_id l_down_x l_down_y l_up_x l_up_y r_down_x r_down_y r_up_x r_up_y
    # pts 顺序: 左下, 左上, 右下, 右上
    pts_flat = [coord for pt in points for coord in pt]
    line = f"{class_id} " + " ".join([f"{x:.1f}" for x in pts_flat]) + "\n"
    
    # 保存
    cv2.imwrite(str(photo_dir / f"{img_idx}.jpg"), frame)
    with open(label_dir / f"{img_idx}.txt", 'w', encoding='utf-8') as f:
        f.write(line)

def annotate_frame(frame, current_class_id):
    """进入单帧的标定界面"""
    points = []
    clone = frame.copy()
    window_name = "Annotate (Esc: Cancel, Enter: Save)"
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((float(x), float(y)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键撤销上一个点
            if points:
                points.pop()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = clone.copy()
        h, w = display.shape[:2]

        # 1. 绘制已标记的点和顺序
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # 颜色区分
        names = ["L-Down", "L-Up", "R-Down", "R-Up"]
        for i, p in enumerate(points):
            cv2.circle(display, (int(p[0]), int(p[1])), 4, colors[i], -1)
            cv2.putText(display, names[i], (int(p[0]) + 5, int(p[1]) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

        # 2. 构建并绘制左下角的实时数据串
        data_str = f"{current_class_id} "
        for i in range(4):
            if i < len(points):
                data_str += f"{points[i][0]:.1f} {points[i][1]:.1f} "
            else:
                data_str += "0.0 0.0 "
        
        cv2.putText(display, data_str.strip(), (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 3. 绘制顶部操作提示 (更新了按键提示)
        cv2.putText(display, f"Class ID: {current_class_id} (0-9 to set, W/S to +/-)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "Left Click: Add Point | Right Click: Undo | C: Clear", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC: 放弃当前帧标定
            cv2.destroyWindow(window_name)
            return False, current_class_id, []
        elif key == ord('c'):  # C: 清空点
            points.clear()
        elif key == 13 or key == 32:  # Enter 或 Space: 确认保存
            if len(points) == 4:
                cv2.destroyWindow(window_name)
                return True, current_class_id, points
            else:
                print("请点满 4 个角点 (左下，左上，右下，右上) 再保存！")
        
        # --- 新增和修改的 ID 切换逻辑 ---
        elif ord('0') <= key <= ord('9'):  
            current_class_id = int(chr(key))
        elif key == ord('w') or key == ord('+') or key == ord('='):  # W 键递增
            current_class_id = min(12, current_class_id + 1)
        elif key == ord('s') or key == ord('-'):                     # S 键递减
            current_class_id = max(0, current_class_id - 1)

def main():
    cfg = load_config()

    if not cfg: 
        print("错误: 无法加载配置文件")
        return

    video_path = Path(cfg.get('video_path', ''))
    export_dir = Path(cfg.get('export_dir', './export'))
    frame_step = cfg.get('frame_step', 5)

    if not video_path.exists():
        print(f"错误: 找不到视频文件 {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("视频无法打开")
        return

    current_class_id = 0
    class_indices = {} 
    
    frame_count = 0
    window_main = "Video Frame (A: Annotate, D/Space: Skip, Q: Quit)"
    cv2.namedWindow(window_main)

    print("--- 启动采样程序 ---")
    print("A 键：对当前帧进行标定")
    print("D 或 空格键：跳过当前帧")
    print("Q 键：退出程序")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放完毕。")
            break

        frame_count += 1
        if frame_count % frame_step != 0:
            continue

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Frame: {frame_count} | Press 'A' to Annotate, 'D' to Skip", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_main, display_frame)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('d') or key == 32:  # Skip
                break
            elif key == ord('a'):  # Annotate
                saved, current_class_id, points = annotate_frame(frame, current_class_id)
                if saved:
                    if current_class_id not in class_indices:
                        class_indices[current_class_id] = get_next_index(export_dir, current_class_id)
                    
                    img_idx = class_indices[current_class_id]
                    save_annotation(export_dir, frame, current_class_id, points, img_idx)
                    
                    print(f"[*] 已保存类别 {current_class_id} 的样本，索引为 {img_idx}")
                    class_indices[current_class_id] += 1
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()