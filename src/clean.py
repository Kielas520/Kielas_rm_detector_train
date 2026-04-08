import math
import shutil
import threading
from pathlib import Path
from queue import Queue
from tqdm import tqdm

def calculate_distance(p1, p2):
    """计算两点之间的欧式距离"""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_frame_center(lines, expected_class_id):
    """
    解析标签并计算当前帧的综合中心点。
    返回: (center_point, status_code)
    status_code: "OK", "ID_ERROR", "FORMAT_ERROR", "EMPTY"
    """
    centers = []
    
    try:
        expected_id_int = int(expected_class_id)
    except ValueError:
        return None, "DIR_ERROR"

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 10:
            continue
            
        try:
            class_id_int = int(parts[0])
        except ValueError:
            return None, "FORMAT_ERROR"

        if class_id_int != expected_id_int:
            return None, "ID_ERROR"

        try:
            l_down_x, l_down_y = float(parts[2]), float(parts[3])
            l_up_x, l_up_y     = float(parts[4]), float(parts[5])
            r_down_x, r_down_y = float(parts[6]), float(parts[7])
            r_up_x, r_up_y     = float(parts[8]), float(parts[9])
            
            center_x = (l_down_x + l_up_x + r_down_x + r_up_x) / 4.0
            center_y = (l_down_y + l_up_y + r_down_y + r_up_y) / 4.0
            centers.append((center_x, center_y))
        except ValueError:
            return None, "FORMAT_ERROR"

    if not centers:
        return None, "EMPTY"

    avg_x = sum(c[0] for c in centers) / len(centers)
    avg_y = sum(c[1] for c in centers) / len(centers)
    return (avg_x, avg_y), "OK"

def io_worker(task_queue, progress_bar):
    """
    后台工作线程：专职处理耗时的文件读写操作
    """
    while True:
        task = task_queue.get()
        if task is None:  # 接收到停止信号
            task_queue.task_done()
            break
            
        out_labels_dir, label_filename, lines, photo_file, out_photos_dir = task
        
        # 1. 写入新的 label 文件
        with open(out_labels_dir / label_filename, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        # 2. 复制图片文件
        shutil.copy2(photo_file, out_photos_dir / photo_file.name)
        
        # 完成一个文件对的处理，更新进度条
        progress_bar.update(1)
        task_queue.task_done()

def clean_dataset_pipeline(raw_dir: str, output_dir: str, distance_threshold: float = 10.0, num_workers: int = 4):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    if not raw_path.exists():
        print(f"错误：找不到原始目录 {raw_path}")
        return

    # 1. 预扫描统计任务总量，准备进度条
    total_files = 0
    valid_class_dirs = []
    for class_dir in raw_path.iterdir():
        if not class_dir.is_dir():
            continue
        labels_dir = class_dir / "labels"
        photos_dir = class_dir / "photos"
        if labels_dir.exists() and photos_dir.exists():
            labels = list(labels_dir.glob("*.txt"))
            total_files += len(labels)
            valid_class_dirs.append((class_dir, labels_dir, photos_dir, labels))

    if total_files == 0:
        print("未发现需要处理的标签文件。")
        return

    # 2. 初始化管道和线程池
    # maxsize=1000 保证内存中最多只缓存 1000 个待处理的数据点，避免爆内存
    io_queue = Queue(maxsize=1000) 
    pbar = tqdm(total=total_files, desc="数据清洗进度", unit="张")
    
    workers = []
    for _ in range(num_workers):
        t = threading.Thread(target=io_worker, args=(io_queue, pbar), daemon=True)
        t.start()
        workers.append(t)

    # 统计指标
    stats = {
        "id_error": 0,
        "distance_skipped": 0,
        "format_error": 0,
        "missing_photo": 0,
        "saved": 0
    }

    # 3. 主线程顺序处理逻辑
    for class_dir, labels_dir, photos_dir, labels in valid_class_dirs:
        expected_class_id = class_dir.name
        out_class_dir = out_path / expected_class_id
        out_labels_dir = out_class_dir / "labels"
        out_photos_dir = out_class_dir / "photos"
        
        out_labels_dir.mkdir(parents=True, exist_ok=True)
        out_photos_dir.mkdir(parents=True, exist_ok=True)

        # 必须排序，保证时序相邻过滤逻辑的正确性
        labels.sort()
        last_saved_center = None

        for label_file in labels:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            current_center, status = get_frame_center(lines, expected_class_id)

            # 异常排错逻辑
            if status == "ID_ERROR":
                stats["id_error"] += 1
                pbar.update(1)
                continue
            elif status != "OK" and status != "EMPTY":
                stats["format_error"] += 1
                pbar.update(1)
                continue

            # 抽样过滤逻辑
            if status == "OK" and last_saved_center is not None and current_center is not None:
                distance = calculate_distance(current_center, last_saved_center)
                if distance < distance_threshold:
                    stats["distance_skipped"] += 1
                    pbar.update(1)
                    continue

            # 寻找对应图片
            photo_file = None
            for ext in ['.jpg', '.png', '.jpeg']:
                temp_photo = photos_dir / (label_file.stem + ext)
                if temp_photo.exists():
                    photo_file = temp_photo
                    break
            
            if photo_file:
                # 校验通过，推送到 IO 队列交由子线程处理
                io_queue.put((out_labels_dir, label_file.name, lines, photo_file, out_photos_dir))
                stats["saved"] += 1
                if current_center is not None:
                    last_saved_center = current_center
            else:
                stats["missing_photo"] += 1
                pbar.update(1)

    # 4. 收尾清理：发送停止信号，等待所有 I/O 任务完成
    for _ in range(num_workers):
        io_queue.put(None)
    for t in workers:
        t.join()

    pbar.close()
    
    # 5. 打印状态报告
    print("\n" + "="*30)
    print("✨ 数据清洗完毕！")
    print("="*30)
    print(f"原始样本总数: {total_files}")
    print(f"成功保留并转移: {stats['saved']}")
    print("-" * 30)
    print("过滤/异常统计:")
    print(f"  ❌ ID 错误数:      {stats['id_error']}")
    print(f"  ⏭️  因重合度跳过数: {stats['distance_skipped']}")
    print(f"  ⚠️  格式/解析错误:  {stats['format_error']}")
    print(f"  🖼️  缺失对应图片:   {stats['missing_photo']}")
    print("="*30)


if __name__ == "__main__":
    clean_dataset_pipeline(
        raw_dir="./data/raw", 
        output_dir="./data/clean", 
        distance_threshold=10.0,
        num_workers=4  # 可根据 CPU 核心数和磁盘性能调整工作线程数
    )