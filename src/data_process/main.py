import sys
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# 导入各个数据处理模块
from src.purify import *
from src.balance import *
from src.augment import *
from src.visiualize import *
from src.split import * # 新增的 split 模块

console = Console()

def load_balance_config(config_path="config.yaml"):
    """尝试加载 balance 的配置参数"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # 根据最新 YAML 结构，如果没有独立 balance 层级，回退到 3000
            return config.get('kielas_rm_train', {}).get('dataset', {}).get('balance', {}).get('max_samples_per_class', 3000)
    except Exception:
        return 3000

def load_split_config(config_path="config.yaml"):
    """尝试加载 split 的验证集比例配置参数"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            val_ratio = float(config.get('kielas_rm_train', {}).get('dataset', {}).get('split', {}).get('val', 0.2))
            return val_ratio
    except Exception:
        return 0.2

def run_purify_step():
    console.print(Panel("开始执行 Purify (数据清洗)", style="cyan"))
    purify_dataset_pipeline(
        raw_dir="./data/raw", 
        output_dir="./data/purify", 
        distance_threshold=10.0,
        num_workers=4
    )

def run_balance_step():
    console.print(Panel("开始执行 Balance (数据均衡)", style="cyan"))
    max_samples = load_balance_config()
    balance_dataset_pipeline(
        input_dir="./data/purify", 
        output_dir="./data/balance", 
        max_samples_per_class=max_samples, 
        num_workers=4
    )

def run_augment_step():
    console.print(Panel("开始执行 Augment (数据增强)", style="cyan"))
    config = AugmentConfig.from_yaml("config.yaml")
    run_augment_pipeline(
        input_dir="./data/balance", 
        output_dir="./data/augment", 
        num_workers=8,
        cfg=config
    )

def run_split_step():
    console.print(Panel("开始执行 Split (数据集拆分)", style="cyan"))
    val_ratio = load_split_config()
    split_dataset_pipeline(
        input_dir="./data/augment", 
        output_dir="./data/datasets", 
        val_ratio=val_ratio,
        num_workers=8
    )

def run_visualize_step(stage: str, if_flag: list):
    console.print(Panel(f"开始执行 Visualize (对 {stage} 阶段抽样可视化)", style="magenta"))
    visualize_dataset(
        root_path="./data", 
        data_type=stage, 
        if_flag=if_flag
    )

def run_full_pipeline():
    """全流程执行：清洗 -> 可视化 -> 均衡 -> 可视化 -> 增强 -> 可视化 -> 拆分 -> 可视化"""
    console.print("\n[bold green]=== 开始全流程数据处理 ===[/bold green]\n")
    
    run_purify_step()
    run_visualize_step("purify", if_flag=[1, 0])
    
    run_balance_step()
    run_visualize_step("balance", if_flag=[0, 0])
    
    run_augment_step()
    run_visualize_step("augment", if_flag=[1, 1])

    run_split_step()
    run_visualize_step("datasets", if_flag=[1, 1])
    
    console.print(Panel("🎉 全流程处理与可视化完成！", border_style="green"))

def interactive_visualize():
    """交互式可视化选择"""
    console.print("\n[bold magenta]请选择要可视化的数据集：[/bold magenta]")
    console.print(" 1. [cyan]raw[/cyan] (原始数据)")
    console.print(" 2. [cyan]purify[/cyan] (清洗后数据)")
    console.print(" 3. [cyan]balance[/cyan] (均衡后数据)")
    console.print(" 4. [cyan]augment[/cyan] (增强后数据)")
    console.print(" 5. [cyan]datasets[/cyan] (拆分后最终数据集)")
    
    sub_choice = Prompt.ask("请输入序号", choices=["1", "2", "3", "4", "5"], default="5")
    
    mapping = {
        "1": ("raw", [1, 0]),
        "2": ("purify", [1, 0]),
        "3": ("balance", [0, 0]),
        "4": ("augment", [1, 1]),
        "5": ("datasets", [1, 1])
    }
    
    stage, flag = mapping[sub_choice]
    run_visualize_step(stage, if_flag=flag)

def main():
    menu_text = (
        "[bold cyan]数据集处理流水线控制台[/bold cyan]\n\n"
        "请输入对应数字序号进行操作:\n"
        "  [bold green]1[/bold green]. 执行 [bold]全流程[/bold] (All-in-one)\n"
        "  [bold green]2[/bold green]. 仅执行 [bold]Purify[/bold] (清洗)\n"
        "  [bold green]3[/bold green]. 仅执行 [bold]Balance[/bold] (均衡)\n"
        "  [bold green]4[/bold green]. 仅执行 [bold]Augment[/bold] (增强)\n"
        "  [bold green]5[/bold green]. 仅执行 [bold]Split[/bold] (拆分)\n"
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
            run_full_pipeline()
            
        elif choice == '2':
            run_purify_step()
            
        elif choice == '3':
            run_balance_step()
            
        elif choice == '4':
            run_augment_step()
            
        elif choice == '5':
            run_split_step()

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