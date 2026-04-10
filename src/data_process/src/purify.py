import math
import shutil
import threading
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.progress import ProgressColumn
from rich.text import Text

console = Console()

class MofNCompleteColumn(ProgressColumn):
    def render(self, task):
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        return Text(f"{completed}/{total}", style="progress.remaining")

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点之间的欧式距离"""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_frame_center(lines: List[str], expected_class_id: str) -> Tuple[Optional[Tuple[float, float]], str]:
    """解析标签并计算当前帧的综合中心点。"""
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
            coords = [float(x) for x in parts[2:10]]
            center_x = sum(coords[0::2]) / 4.0
            center_y = sum(coords[1::2]) / 4.0
            centers.append((center_x, center_y))
        except ValueError:
            return None, "FORMAT_ERROR"

    if not centers:
        return None, "EMPTY"

    avg_x = sum(c[0] for c in centers) / len(centers)
    avg_y = sum(c[1] for c in centers) / len(centers)
    return (avg_x, avg_y), "OK"

def io_worker(task_queue: Queue, progress: Progress, task_id):
    """后台工作线程：专职处理耗时的文件读写操作"""
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break
            
        out_labels_dir, label_filename, lines, photo_file, out_photos_dir = task
        
        with open(out_labels_dir / label_filename, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        shutil.copy2(photo_file, out_photos_dir / photo_file.name)
        
        progress.advance(task_id)
        task_queue.task_done()

def purify_dataset_pipeline(raw_dir: str, output_dir: str, distance_threshold: float = 10.0, num_workers: int = 4):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    if not raw_path.exists():
        console.print(f"[bold red]错误：[/bold red]找不到原始目录 {raw_path}")
        return

    # ==========================================
    # 新增：检测并刷新输出文件夹
    # ==========================================
    if out_path.exists():
        console.print(f"[yellow]检测到输出目录已存在，正在清理旧数据: {out_path}[/yellow]")
        shutil.rmtree(out_path)
    
    # 重新创建干净的输出主目录
    out_path.mkdir(parents=True, exist_ok=True)
    # ==========================================

    # 1. 预扫描
    valid_class_dirs = []
    total_files = 0
    
    with console.status("[bold green]正在扫描目录..."):
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
        console.print("[yellow]未发现需要处理的标签文件。[/yellow]")
        return

    # 2. 初始化 Rich 进度条
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console
    )

    stats = {
        "id_error": 0, 
        "distance_skipped": 0, 
        "format_error": 0, 
        "missing_photo": 0, 
        "empty_label": 0, 
        "saved": 0
    }
    io_queue = Queue(maxsize=1000)

    with progress:
        main_task = progress.add_task("[cyan]数据清洗进度", total=total_files)
        
        workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=io_worker, args=(io_queue, progress, main_task), daemon=True)
            t.start()
            workers.append(t)

        # 3. 主逻辑
        for class_dir, labels_dir, photos_dir, labels in valid_class_dirs:
            expected_class_id = class_dir.name
            out_class_dir = out_path / expected_class_id
            out_labels_dir = out_class_dir / "labels"
            out_photos_dir = out_class_dir / "photos"
            
            out_labels_dir.mkdir(parents=True, exist_ok=True)
            out_photos_dir.mkdir(parents=True, exist_ok=True)

            labels.sort()
            last_saved_center = None

            for label_file in labels:
                # 检查图片是否存在
                photo_file = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    temp_photo = photos_dir / (label_file.stem + ext)
                    if temp_photo.exists():
                        photo_file = temp_photo
                        break
                
                if not photo_file:
                    stats["missing_photo"] += 1
                    progress.advance(main_task)
                    continue

                # 读取标签文件
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 拦截全空文件或只包含空白符的文件
                if not lines or all(line.strip() == '' for line in lines):
                    stats["empty_label"] += 1
                    progress.advance(main_task)
                    continue

                current_center, status = get_frame_center(lines, expected_class_id)

                if status == "EMPTY":
                    stats["empty_label"] += 1
                    progress.advance(main_task)
                    continue
                elif status == "ID_ERROR":
                    stats["id_error"] += 1
                    progress.advance(main_task)
                    continue
                elif status != "OK":
                    stats["format_error"] += 1
                    progress.advance(main_task)
                    continue

                if last_saved_center is not None:
                    distance = calculate_distance(current_center, last_saved_center)
                    if distance < distance_threshold:
                        stats["distance_skipped"] += 1
                        progress.advance(main_task)
                        continue

                # 验证均通过，推入队列保存
                io_queue.put((out_labels_dir, label_file.name, lines, photo_file, out_photos_dir))
                stats["saved"] += 1
                last_saved_center = current_center

        # 4. 收尾
        for _ in range(num_workers):
            io_queue.put(None)
        for t in workers:
            t.join()

    # 5. 打印报告
    print_report(total_files, stats)

def print_report(total, stats):
    table = Table(title="数据清洗任务报告", show_header=True, header_style="bold magenta")
    table.add_column("统计项", style="dim")
    table.add_column("数量", justify="right")
    table.add_column("占比", justify="right")

    def add_row(name, value, color="white"):
        percent = f"{(value/total)*100:.1f}%" if total > 0 else "0%"
        table.add_row(name, f"[{color}]{value}[/{color}]", percent)

    add_row("成功保留", stats['saved'], "green")
    add_row("ID 匹配错误", stats['id_error'], "red")
    add_row("距离过滤 (跳过)", stats['distance_skipped'], "yellow")
    add_row("格式解析错误", stats['format_error'], "red")
    add_row("图片缺失 (被忽略)", stats['missing_photo'], "red")
    add_row("空标签 (被忽略)", stats['empty_label'], "red")

    console.print("\n")
    console.print(Panel(table, expand=False, border_style="cyan", title="✨ 清洗完成"))

if __name__ == "__main__":
    purify_dataset_pipeline(
        raw_dir="./data/raw", 
        output_dir="./data/purify", 
        distance_threshold=10.0,
        num_workers=4
    )