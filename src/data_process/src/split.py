import random
import shutil
import threading
from pathlib import Path
from queue import Queue
import yaml

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
    后台 I/O 线程：负责将图片和标签拷贝到最终的训练/验证集中
    """
    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break
            
        src_photo, dst_photo, src_label, dst_label = task
        
        try:
            if src_photo and src_photo.exists():
                shutil.copy2(src_photo, dst_photo)
            if src_label and src_label.exists():
                shutil.copy2(src_label, dst_label)
        except Exception as e:
            console.print(f"[red]文件拷贝错误 {src_photo.name}: {e}[/red]")
            
        progress.advance(task_id)
        task_queue.task_done()

def generate_yaml(input_dir: Path, output_dir: Path):
    """继承上一步的 yaml，并生成新的标准数据集 yaml"""
    old_yaml_path = input_dir / "train.yaml"
    new_yaml_path = output_dir / "dataset.yaml"
    
    # 默认配置（以防旧 yaml 不存在）
    nc = 0
    names_dict = {}
    weights_dict = {}
    
    if old_yaml_path.exists():
        try:
            with open(old_yaml_path, 'r', encoding='utf-8') as f:
                old_cfg = yaml.safe_load(f)
                nc = old_cfg.get('nc', 0)
                names_dict = old_cfg.get('names', {})
                weights_dict = old_cfg.get('weights', {})
        except Exception as e:
            console.print(f"[red]读取旧 yaml 失败: {e}[/red]")
            
    # 按照旧代码风格手动拼接 YAML 保证格式清晰
    yaml_content = f"""# RM Target Detection Dataset Configuration
path: {output_dir.absolute()}
train: images/train
val: images/val

nc: {nc}

names:
"""
    # 写入类别名
    sorted_cids = sorted(names_dict.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    for cid in sorted_cids:
        yaml_content += f"  {cid}: '{names_dict[cid]}'\n"

    # 如果有权重，也继承过来
    if weights_dict:
        yaml_content += "\nweights:\n"
        for cid in sorted_cids:
            weight = weights_dict.get(cid, 1.0)
            yaml_content += f"  {cid}: {weight:.4f}\n"

    with open(new_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

def split_dataset_pipeline(input_dir: str, output_dir: str, val_ratio: float = 0.2, num_workers: int = 8):
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists():
        console.print(f"[bold red]错误：[/bold red]找不到输入目录 {in_path}")
        return

    # 1. 初始化输出目录 (标准格式: images/train, images/val, labels/train, labels/val)
    if out_path.exists():
        console.print(f"[yellow]检测到输出目录已存在，正在清理旧数据: {out_path}[/yellow]")
        shutil.rmtree(out_path)

    for split_type in ["train", "val"]:
        (out_path / "images" / split_type).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split_type).mkdir(parents=True, exist_ok=True)

    # 2. 扫描数据并制定划分策略
    all_tasks = []
    
    table = Table(title="数据集拆分统计", header_style="bold blue")
    table.add_column("类别 ID", justify="center")
    table.add_column("总数量", justify="right")
    table.add_column("Train", justify="right", style="green")
    table.add_column("Val", justify="right", style="magenta")

    with console.status("[bold green]正在扫描数据并分配划分队列..."):
        for class_dir in in_path.iterdir():
            if not class_dir.is_dir(): continue
            
            class_id = class_dir.name
            labels_dir = class_dir / "labels"
            photos_dir = class_dir / "photos"

            if not labels_dir.exists(): continue

            # 收集该类别的所有有效配对
            class_pairs = []
            for label_file in labels_dir.glob("*.txt"):
                photo_file = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    temp_photo = photos_dir / (label_file.stem + ext)
                    if temp_photo.exists():
                        photo_file = temp_photo
                        break
                if photo_file:
                    class_pairs.append((label_file, photo_file))

            if not class_pairs: continue

            # 打乱并拆分
            random.shuffle(class_pairs)
            val_count = int(len(class_pairs) * val_ratio)
            train_count = len(class_pairs) - val_count
            
            table.add_row(class_id, str(len(class_pairs)), str(train_count), str(val_count))

            # 分配任务 (添加前缀防止不同类别下同名文件覆盖)
            for i, (label_file, photo_file) in enumerate(class_pairs):
                split_type = "val" if i < val_count else "train"
                
                # 构造新的安全文件名
                safe_stem = f"{class_id}_{label_file.stem}"
                dst_label = out_path / "labels" / split_type / f"{safe_stem}.txt"
                dst_photo = out_path / "images" / split_type / f"{safe_stem}{photo_file.suffix}"
                
                all_tasks.append((photo_file, dst_photo, label_file, dst_label))

    if not all_tasks:
        console.print("[yellow]未扫描到有效数据文件。[/yellow]")
        return

    console.print(table)
    console.print(f"\n[bold]总拆分文件数:[/bold] [yellow]{len(all_tasks)}[/yellow] (拆分比例 Train:{1-val_ratio:.2f} / Val:{val_ratio:.2f})\n")

    # 3. 继承并生成 YAML
    generate_yaml(in_path, out_path)

    # 4. 执行多线程拷贝
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console
    )

    with progress:
        main_task = progress.add_task("[cyan]构建最终数据集...", total=len(all_tasks))
        io_queue = Queue(maxsize=2000)
        
        workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=io_worker, args=(io_queue, progress, main_task), daemon=True)
            t.start()
            workers.append(t)

        # 派发任务
        for task in all_tasks:
            io_queue.put(task)

        # 结束信号
        for _ in range(num_workers):
            io_queue.put(None)
            
        for t in workers:
            t.join()

    console.print(Panel(
        f"✨ [bold green]数据集准备完毕！[/bold green]\n\n"
        f"输出路径: [underline]{out_path.absolute()}[/underline]\n"
        f"配置清单: [cyan]dataset.yaml[/cyan]\n"
        f"目前可直接将 dataset.yaml 用于模型训练。", 
        border_style="green"
    ))

if __name__ == "__main__":
    config_path = "config.yaml"
    val_ratio = 0.2  # 默认值
    
    # 解析 YAML 配置文件
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
            # 安全地逐层获取嵌套的参数
            split_cfg = config_data.get('kielas_rm_train', {}).get('dataset', {}).get('split', {})
            if 'val' in split_cfg:
                val_ratio = float(split_cfg['val'])
                console.print(f"[green]已从 {config_path} 加载拆分配置，验证集比例: {val_ratio}[/green]")
            else:
                console.print(f"[yellow]YAML 中未发现 kielas_rm_train.dataset.split.val 配置，使用默认比例 {val_ratio}[/yellow]")
    except Exception as e:
        console.print(f"[red]读取或解析 {config_path} 失败: {e}，将使用默认比例 {val_ratio}[/red]")

    # 执行拆分管线
    split_dataset_pipeline(
        input_dir="./data/augment", 
        output_dir="./data/datasets", 
        val_ratio=val_ratio,
        num_workers=8
    )