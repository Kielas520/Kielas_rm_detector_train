import sys
import yaml
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# 导入各个数据处理模块
from src.data_process.src import *

console = Console()

def load_yaml_config(config_path="config.yaml"):
    """加载并解析 YAML 配置文件"""
    p = Path(config_path)
    if not p.exists():
        console.print(f"[red]错误: 找不到配置文件 {config_path}[/red]")
        return {}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config if isinstance(config, dict) else {}
    except Exception as e:
        console.print(f"[red]读取配置文件出错: {e}[/red]")
        return {}

def get_path(config, key, default):
    """
    统一路径获取函数：优先从 config.yaml 获取，自动展开 ~ 符号，返回 pathlib.Path
    """
    dataset_cfg = config.get('kielas_rm_train', {}).get('dataset', {})
    path_str = dataset_cfg.get(key, default)
    return Path(os.path.expanduser(str(path_str)))

def check_dir(path: Path, must_exist=True):
    """检查目录是否存在"""
    if must_exist and not path.exists():
        console.print(f"[red]错误：找不到目录 {path}[/red]")
        return False
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return True

def run_purify_step(config):
    raw_dir = get_path(config, 'raw_dir', "./data/raw")
    purify_dir = get_path(config, 'purify_dir', "./data/purify")
    
    if not check_dir(raw_dir, must_exist=True): return
    check_dir(purify_dir, must_exist=False)

    console.print(Panel(f"开始执行 Purify (数据清洗)\n输入路径: {raw_dir}\n输出路径: {purify_dir}", style="cyan"))
    # 恢复执行函数，并将 Path 对象转为 str 传入
    purify_dataset_pipeline(
        raw_dir=str(raw_dir), 
        output_dir=str(purify_dir), 
        distance_threshold=10.0,
        num_workers=4
    )

def run_balance_step(config):
    purify_dir = get_path(config, 'purify_dir', "./data/purify")
    balance_dir = get_path(config, 'balance_dir', "./data/balance")
    max_samples = config.get('kielas_rm_train', {}).get('dataset', {}).get('balance', {}).get('max_samples_per_class', 3000)

    if not check_dir(purify_dir, must_exist=True): return
    check_dir(balance_dir, must_exist=False)

    console.print(Panel(f"开始执行 Balance (数据均衡)\n输入路径: {purify_dir}\n输出路径: {balance_dir}", style="cyan"))
    # 恢复执行函数
    balance_dataset_pipeline(
        input_dir=str(purify_dir), 
        output_dir=str(balance_dir), 
        max_samples_per_class=max_samples, 
        num_workers=4
    )

def run_split_step(config):
    balance_dir = get_path(config, 'balance_dir', "./data/balance")
    datasets_dir = get_path(config, 'datasets_dir', "./data/datasets")
    val_ratio = float(config.get('kielas_rm_train', {}).get('dataset', {}).get('split', {}).get('val', 0.2))

    if not check_dir(balance_dir, must_exist=True): return
    check_dir(datasets_dir, must_exist=False)

    console.print(Panel(f"开始执行 Split (数据集拆分)\n输入路径: {balance_dir}\n输出路径: {datasets_dir}", style="cyan"))
    # 恢复执行函数
    split_dataset_pipeline(
        input_dir=str(balance_dir), 
        output_dir=str(datasets_dir), 
        val_ratio=val_ratio,
        num_workers=8
    )

def run_augment_step(config):
    datasets_dir = get_path(config, 'datasets_dir', "./data/datasets")
    if not check_dir(datasets_dir, must_exist=True): return

    console.print(Panel(f"开始执行 Augment (数据增强 - 仅针对训练集)\n目标路径: {datasets_dir}", style="cyan"))
    # 恢复执行函数
    aug_cfg = AugmentConfig.from_yaml("config.yaml")
    run_augment_pipeline(
        dataset_dir=str(datasets_dir), 
        num_workers=8,
        cfg=aug_cfg
    )

def run_visualize_step(stage: str, if_flag: list):
    root_path = Path("./data")
    console.print(Panel(f"开始执行 Visualize (对 {stage} 阶段抽样可视化)", style="magenta"))
    # 恢复执行函数
    visualize_dataset(
        root_path=str(root_path), 
        data_type=stage, 
        if_flag=if_flag
    )

def run_full_pipeline(config):
    """全流程执行：清洗 -> 均衡 -> 拆分 -> 增强 -> 可视化"""
    console.print("\n[bold green]=== 开始全流程数据处理 ===[/bold green]\n")
    
    run_purify_step(config)
    run_visualize_step("purify", if_flag=[1, 0])
    
    run_balance_step(config)
    run_visualize_step("balance", if_flag=[0, 0])
    
    run_split_step(config)
    
    run_augment_step(config)
    
    run_visualize_step("datasets", if_flag=[1, 1])
    
    console.print(Panel("🎉 全流程处理与可视化完成！", border_style="green"))

def interactive_visualize():
    """交互式可视化选择"""
    console.print("\n[bold magenta]请选择要可视化的数据集：[/bold magenta]")
    console.print(" 1. [cyan]raw[/cyan] (原始数据)")
    console.print(" 2. [cyan]purify[/cyan] (清洗后数据)")
    console.print(" 3. [cyan]balance[/cyan] (均衡后数据)")
    console.print(" 4. [cyan]datasets[/cyan] (最终数据集 - 包含增强后的训练集)")
    
    sub_choice = Prompt.ask("请输入序号", choices=["1", "2", "3", "4"], default="4")
    
    mapping = {
        "1": ("raw", [1, 0]),
        "2": ("purify", [1, 0]),
        "3": ("balance", [0, 0]),
        "4": ("datasets", [1, 1])
    }
    
    stage, flag = mapping[sub_choice]
    run_visualize_step(stage, if_flag=flag)

def main():
    config = load_yaml_config("config.yaml")
    
    menu_text = (
        "[bold cyan]数据集处理流水线控制台[/bold cyan]\n\n"
        "请输入对应数字序号进行操作:\n"
        "  [bold green]1[/bold green]. 执行 [bold]全流程[/bold] (All-in-one)\n"
        "  [bold green]2[/bold green]. 仅执行 [bold]Purify[/bold] (清洗)\n"
        "  [bold green]3[/bold green]. 仅执行 [bold]Balance[/bold] (均衡)\n"
        "  [bold green]4[/bold green]. 仅执行 [bold]Split[/bold] (拆分)\n"
        "  [bold green]5[/bold green]. 仅执行 [bold]Augment[/bold] (增强训练集)\n"
        "  [bold green]6[/bold green]. 执行 [bold]Visualize[/bold] (可视化特定阶段)\n"
        "  [bold red]0[/bold red]. 退出程序"
    )
    
    console.print(Panel.fit(menu_text, border_style="cyan"))

    while True:
        choice = Prompt.ask("\n请选择操作序号", choices=["1", "2", "3", "4", "5", "6", "0"], default="1")

        if choice == '0':
            console.print("[yellow]已退出。[/yellow]")
            break
            
        elif choice == '1':
            run_full_pipeline(config)
            
        elif choice == '2':
            run_purify_step(config)
            
        elif choice == '3':
            run_balance_step(config)
            
        elif choice == '4':
            run_split_step(config)
            
        elif choice == '5':
            run_augment_step(config)

        elif choice == '6':
            interactive_visualize()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]程序被用户中断。[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]程序运行出错: {e}[/red]")
        sys.exit(1)