import sys
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# 导入各个数据处理模块 (注意保持与实际文件名一致)
import purify
import balance
import augment
import visiualize  # 按照你上传的文件名拼写

console = Console()

def load_balance_config(config_path="config.yaml"):
    """尝试加载 balance 的配置参数"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('balance', {}).get('max_samples_per_class', 3000)
    except Exception:
        return 3000

def run_purify_step():
    console.print(Panel("开始执行 Purify (数据清洗)", style="cyan"))
    purify.purify_dataset_pipeline(
        raw_dir="./data/raw", 
        output_dir="./data/purify", 
        distance_threshold=10.0,
        num_workers=4
    )

def run_balance_step():
    console.print(Panel("开始执行 Balance (数据均衡)", style="cyan"))
    max_samples = load_balance_config()
    balance.balance_dataset_pipeline(
        input_dir="./data/purify", 
        output_dir="./data/balance", 
        max_samples_per_class=max_samples, 
        num_workers=4
    )

def run_augment_step():
    console.print(Panel("开始执行 Augment (数据增强)", style="cyan"))
    config = augment.AugmentConfig.from_yaml("config.yaml")
    augment.run_augment_pipeline(
        input_dir="./data/balance", 
        output_dir="./data/augment", 
        num_workers=8,
        cfg=config
    )

def run_visualize_step(stage: str, if_flag: list):
    console.print(Panel(f"开始执行 Visualize (对 {stage} 阶段抽样可视化)", style="magenta"))
    visiualize.visualize_dataset(
        root_path="./data", 
        data_type=stage, 
        if_flag=if_flag
    )

def run_full_pipeline():
    """全流程执行：清洗 -> 可视化 -> 均衡 -> 可视化 -> 增强 -> 可视化"""
    console.print("\n[bold green]=== 开始全流程数据处理 ===[/bold green]\n")
    
    # 1. Purify
    run_purify_step()
    # Purify 输出10个数据位 (包含颜色)，颜色 flag_type=0
    run_visualize_step("purify", if_flag=[1, 0])
    
    # 2. Balance
    run_balance_step()
    # Balance 剔除了颜色位，剩下9个数据位，无 flag
    run_visualize_step("balance", if_flag=[0, 0])
    
    # 3. Augment
    run_augment_step()
    # Augment 添加了可见度位，输出10个数据位，可见度 flag_type=1
    run_visualize_step("augment", if_flag=[1, 1])
    
    console.print(Panel("🎉 全流程处理与可视化完成！", border_style="green"))

def main():
    console.print(Panel.fit(
        "[bold cyan]数据集处理流水线控制台[/bold cyan]\n\n"
        "支持的指令:\n"
        "  [green]all[/green]       - 执行全流程 (自动包含每次操作后的可视化)\n"
        "  [green]purify[/green]    - 仅执行数据清洗\n"
        "  [green]balance[/green]   - 仅执行数据均衡\n"
        "  [green]augment[/green]   - 仅执行数据增强\n"
        "  [green]visualize[/green] - 仅执行数据抽样可视化\n"
        "  [red]exit[/red]      - 退出程序",
        border_style="cyan"
    ))

    while True:
        choice = Prompt.ask("\n请输入模式名称", default="all").strip().lower()

        if choice == 'exit' or choice == 'q':
            console.print("[yellow]已退出。[/yellow]")
            sys.exit(0)
            
        elif choice == 'all':
            run_full_pipeline()
            
        elif choice == 'purify':
            run_purify_step()
            
        elif choice == 'balance':
            run_balance_step()
            
        elif choice == 'augment':
            run_augment_step()
            
        elif choice == 'visualize' or choice == 'visiualize':
            stage = Prompt.ask("请输入要可视化的文件夹名称 (如 raw, purify, balance, augment)", default="augment")
            # 简单推断一下 flag，如果不确定默认使用 [0, 0]
            if stage == "purify":
                flag = [1, 0]
            elif stage == "augment":
                flag = [1, 1]
            else:
                flag = [0, 0]
            run_visualize_step(stage, if_flag=flag)
            
        else:
            console.print(f"[red]无效的模式: {choice}，请重新输入。[/red]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]程序被用户中断。[/yellow]")
        sys.exit(0)