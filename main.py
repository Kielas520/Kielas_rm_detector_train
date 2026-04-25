import sys
import subprocess
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

console = Console()

class WorkflowTerminal:
    def __init__(self):
        # 获取当前根目录路径
        self.root_path = Path(__file__).parent.absolute()
        
        self.menu_options = {
            "1": {"desc": "数据预处理 (Purify/Balance/Split/Augment)", "module": "src.data_process.process"},
            "2": {"desc": "开启模型训练 (Train & Evaluate)", "module": "src.training.train"},
            "3": {"desc": "模型格式导出 (Export to ONNX/TorchScript)", "module": "src.training.export"},
            "4": {"desc": "实时推理演示 (Camera/Video Demo)", "module": "src.demo.demo"},
            "0": {"desc": "退出系统", "module": None}
        }

    def display_menu(self):
        table = Table(show_header=False, border_style="cyan", box=None)
        table.add_column("Index", style="bold green", justify="right")
        table.add_column("Description", style="white")

        for key, value in self.menu_options.items():
            table.add_row(key, value["desc"])

        menu_panel = Panel(
            table,
            title="[bold magenta]RoboMaster 视觉神经网络菜单[/bold magenta]",
            subtitle="[gray]Subprocess Execution Mode[/gray]",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(menu_panel)

    def run_script(self, module_name):
        """使用 subprocess -m 模式调用模块"""
        try:
            # 改用 Popen，这样我们能灵活控制父进程的等待逻辑
            process = subprocess.Popen([sys.executable, "-m", module_name])
            process.wait()
        except KeyboardInterrupt:
            # 按下 Ctrl+C 时，父子进程会同时收到信号。
            # 子进程 (train.py) 此时已经触发了 __exit__ 正在疯狂保存。
            # 父进程 (main.py) 收到信号后进入这个 except 块，在这里老老实实等子进程存完。
            console.print("\n[yellow]⏳ 检测到中断，正在等待训练进程保存权重与可视化，请稍候...[/yellow]")
            try:
                process.wait()  # 再次等待，直到子进程安全走完流程退出
            except KeyboardInterrupt:
                # 如果你急眼了，又连按了一次 Ctrl+C，那就连保存都不管了，直接强杀
                console.print("\n[red]💥 连续中断，强制击杀进程！[/red]")
                process.kill()
                
        if process.returncode != 0 and process.returncode is not None:
            console.print(f"[dim]独立进程已结束 (退出码: {process.returncode})[/dim]")

    def run(self):
        while True:
            console.clear()
            self.display_menu()
            
            choice = Prompt.ask(
                "\n[bold cyan]请选择操作序号[/bold cyan]", 
                choices=list(self.menu_options.keys()), 
                default="1"
            )

            if choice == "0":
                console.print("[yellow]正在退出系统... 祝调试顺利！[/yellow]")
                break

            # 执行对应脚本
            selected_module = self.menu_options[choice]["module"]
            console.print(f"\n[bold reverse] 正在启动独立进程: {self.menu_options[choice]['desc']} [/bold reverse]\n")
            
            self.run_script(selected_module)
            
            Prompt.ask("\n[dim]按回车键返回主菜单...[/dim]")

if __name__ == "__main__":
    terminal = WorkflowTerminal()
    terminal.run()