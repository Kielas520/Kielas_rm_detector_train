import shutil
import threading
import random
from pathlib import Path
from queue import Queue
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import yaml

# 引入 rich 组件
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, ProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

class MofNCompleteColumn(ProgressColumn):
    """自定义列显示 n/m 格式"""
    def render(self, task):
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        return Text(f"{completed}/{total}", style="progress.remaining")

def io_worker(task_queue: Queue, progress: Progress, task_id):
    """
    后台 I/O 线程：负责读取旧标签、剔除 color 字段、写入新标签以及拷贝图片
    """
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break
            
        in_label, out_label, in_photo, out_photo = task
        
        # 1. 重新格式化标签：剔除 color (索引为 1 的列)
        try:
            with open(in_label, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 10: 
                    parts.pop(1)  # 移除 color 列
                    new_lines.append(" ".join(parts) + "\n")
                
            with open(out_label, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
                
            # 2. 拷贝图片文件
            if in_photo and in_photo.exists() and out_photo:
                shutil.copy2(in_photo, out_photo)
        except Exception as e:
            console.print(f"[red]错误处理文件 {in_label.name}: {e}[/red]")
            
        progress.advance(task_id)
        task_queue.task_done()

def generate_yaml(output_dir: Path, class_counts: dict, class_weights: dict):
    """生成类似 YOLO 的 train.yaml 配置文件"""
    yaml_path = output_dir / "train.yaml"
    sorted_cids = sorted(class_counts.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    yaml_content = f"""# Train Dataset Configuration
path: {output_dir.absolute()}
train: ./
val: ./

nc: {len(class_counts)}

names:
"""
    for cid in sorted_cids:
        yaml_content += f"  {cid}: '{cid}'\n"

    yaml_content += "\nweights:\n"
    for cid in sorted_cids:
        yaml_content += f"  {cid}: {class_weights[cid]:.4f}\n"

    yaml_content += """
features_description:
  - class_id
  - left_light_down_x
  - left_light_down_y
  - left_light_up_x
  - left_light_up_y
  - right_light_down_x
  - right_light_down_y
  - right_light_up_x
  - right_light_up_y
  - center_x
  - center_y
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

def balance_dataset_pipeline(input_dir: str, output_dir: str, max_samples_per_class: int = 3000, num_workers: int = 4):
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists():
        console.print(f"[bold red]错误：[/bold red]找不到输入目录 {in_path}")
        return

    # 1. 扫描与统计
    class_files = defaultdict(list)
    with console.status("[bold green]正在扫描原始数据并分析类别分布..."):
        for class_dir in in_path.iterdir():
            if not class_dir.is_dir(): continue
            
            class_id = class_dir.name
            labels_dir = class_dir / "labels"
            photos_dir = class_dir / "photos"

            if not labels_dir.exists(): continue

            for label_file in labels_dir.glob("*.txt"):
                photo_file = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    temp_photo = photos_dir / (label_file.stem + ext)
                    if temp_photo.exists():
                        photo_file = temp_photo
                        break
                class_files[class_id].append((label_file, photo_file))

    if not class_files:
        console.print("[yellow]未扫描到有效数据文件。[/yellow]")
        return

    # 2. 下采样策略预览
    selected_files = {}
    strategy_table = Table(title="数据集平衡策略预览", header_style="bold magenta")
    strategy_table.add_column("类别 ID", justify="center")
    strategy_table.add_column("原始数量", justify="right")
    strategy_table.add_column("下采样后", justify="right", style="green")
    strategy_table.add_column("训练权重", justify="right", style="cyan")

    for cid in sorted(class_files.keys(), key=lambda x: int(x) if x.isdigit() else x):
        files = class_files[cid]
        if len(files) > max_samples_per_class:
            selected_files[cid] = random.sample(files, max_samples_per_class)
        else:
            selected_files[cid] = files
        
        orig_cnt = len(files)
        new_cnt = len(selected_files[cid])
        # 暂时记录，稍后计算权重
        strategy_table.add_row(cid, str(orig_cnt), str(new_cnt), "CALC...")

    class_counts = {cid: len(f) for cid, f in selected_files.items()}
    max_count = max(class_counts.values())
    class_weights = {cid: max_count / count for cid, count in class_counts.items()}

    # 更新表格中的权重列
    for i, cid in enumerate(sorted(class_counts.keys(), key=lambda x: int(x) if x.isdigit() else x)):
        strategy_table.columns[3]._cells[i] = f"{class_weights[cid]:.4f}"

    console.print(strategy_table)
    console.print(f"\n[bold]总处理任务数:[/bold] [yellow]{sum(class_counts.values())}[/yellow]\n")

    # 3. 创建输出并生成 YAML
    out_path.mkdir(parents=True, exist_ok=True)
    generate_yaml(out_path, class_counts, class_weights)

    # 4. 执行多线程转换
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console
    )

    with progress:
        main_task = progress.add_task("[cyan]转换与转移进度", total=sum(class_counts.values()))
        io_queue = Queue(maxsize=1000)
        
        workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=io_worker, args=(io_queue, progress, main_task), daemon=True)
            t.start()
            workers.append(t)

        for cid, files in selected_files.items():
            out_class_dir = out_path / cid
            (out_class_dir / "labels").mkdir(parents=True, exist_ok=True)
            (out_class_dir / "photos").mkdir(parents=True, exist_ok=True)

            for label_file, photo_file in files:
                out_label = out_class_dir / "labels" / label_file.name
                out_photo = out_class_dir / "photos" / photo_file.name if photo_file else None
                io_queue.put((label_file, out_label, photo_file, out_photo))

        for _ in range(num_workers):
            io_queue.put(None)
        for t in workers:
            t.join()

    console.print(Panel(f"✨ [bold green]处理完毕！[/bold green]\n\n数据已保存至: [underline]{out_path.absolute()}[/underline]\n配置文件: [cyan]train.yaml[/cyan]", border_style="green"))

if __name__ == "__main__":
    balance_dataset_pipeline(
        input_dir="./data/purify", 
        output_dir="./data/balance", 
        max_samples_per_class=3000, 
        num_workers=4
    )