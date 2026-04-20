import cv2
import yaml
import numpy as np
from pathlib import Path

def main():
    # 1. 读取 YAML 配置
    config_path = Path("config.yaml")
    if not config_path.exists():
        print(f"错误：找不到配置文件 {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 假设你的 yaml 结构是 kielas_rm_train -> train -> input_size
    try:
        train_cfg = config['kielas_rm_train']['train']
        input_size = train_cfg.get('input_size', [416, 416])
        target_w, target_h = input_size[0], input_size[1]
    except KeyError:
        print("YAML 格式不符合预期，请检查路径：kielas_rm_train -> train -> input_size")
        target_w, target_h = 416, 416

    # 2. 初始化相机
    cap = cv2.VideoCapture(0) # 如果有多个相机，请尝试 1, 2...
    if not cap.isOpened():
        print("无法打开相机")
        return

    # 3. 创建窗口和滑动条
    window_name = "Pixel Size Helper"
    cv2.namedWindow(window_name)
    
    # 滑动条回调函数（此处不需要做额外操作）
    def nothing(x):
        pass

    # 创建滑动条，最大值为输入尺寸的宽度
    cv2.createTrackbar("Square Size", window_name, 32, target_w, nothing)

    print(f"--- 启动成功 ---")
    print(f"目标分辨率: {target_w}x{target_h}")
    print(f"操作提示: 拖动滑动条改变正方形大小，按 'q' 键退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. 压缩画面到输入 size
        img_resized = cv2.resize(frame, (target_w, target_h))
        
        # 5. 获取滑动条数值
        side_len = cv2.getTrackbarPos("Square Size", window_name)
        if side_len < 1: side_len = 1 # 防止为0

        # 6. 计算正方形坐标 (居中)
        cx, cy = target_w // 2, target_h // 2
        x1 = cx - side_len // 2
        y1 = cy - side_len // 2
        x2 = x1 + side_len
        y2 = y1 + side_len

        # 7. 绘制
        # 绘制中心正方形 (红色)
        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 绘制对角线辅助定位
        cv2.line(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.line(img_resized, (x1, y2), (x2, y1), (0, 255, 0), 1)

        # 实时显示像素宽度信息
        text = f"Width: {side_len} px"
        cv2.putText(img_resized, text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示提示信息
        cv2.putText(img_resized, f"Input Size: {target_w}x{target_h}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 8. 显示结果
        cv2.imshow(window_name, img_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()