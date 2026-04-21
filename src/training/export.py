import yaml
import torch
from pathlib import Path
from rich.console import Console
from rich.status import Status
from rich.panel import Panel

from src.training.src import RMDetector

# 初始化 rich 终端控制台
console = Console()

def export_onnx(model, dummy_input, output_path: Path, cfg):
    """导出 ONNX 格式并根据配置进行轻量化"""
    console.print(f"[*] 开始导出 ONNX 模型: [cyan]{output_path}[/cyan]")
    
    simplify = cfg['onnx'].get('simplify', True)
    opset_version = cfg['onnx'].get('opset', 18)
    
    with Status("[bold yellow]正在导出原生 ONNX 模型 (Legacy TorchScript 引擎)...", console=console):
        # 核心修复点：
        # 1. 传入原始 model，不使用 torch.jit.trace
        # 2. 显式追加 dynamo=False 参数，强制规避 PyTorch 2.5+ 不稳定的底层引擎
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            # --- 修改点 1: 适配多尺度，提供 3 个输出节点名称 ---
            output_names=['output_p3', 'output_p4', 'output_p5'], 
            dynamo=False
        )
    
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            with Status("[bold yellow]正在执行 ONNX 极致轻量化 (onnxsim)...", console=console):
                onnx_model = onnx.load(str(output_path))
                model_simp, check = onnx_simplify(onnx_model)
                
                if check:
                    onnx.save(model_simp, str(output_path))
                    console.print(f"[+] [bold green]ONNX 轻量化完成[/bold green]，文件已保存至: [cyan]{output_path}[/cyan]")
                else:
                    console.print("[-] [bold red]ONNX 轻量化校验失败，保留初始版本。[/bold red]")
        except ImportError:
            console.print("[-] [bold red]未检测到 onnx 或 onnxsim 库，跳过极致轻量化步骤。[/bold red]")
            console.print("    建议执行: [white]pip install onnx onnxsim[/white]")
    else:
        console.print(f"[+] [bold green]ONNX 导出完成（按配置跳过轻量化）[/bold green]，文件已保存至: [cyan]{output_path}[/cyan]")

def export_torchscript(model, dummy_input, output_path: Path):
    """导出 TorchScript 格式"""
    console.print(f"[*] 开始导出 TorchScript 模型: [cyan]{output_path}[/cyan]")
    with Status("[bold yellow]正在跟踪生成 TorchScript 模型...", console=console):
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(output_path)) # type: ignore
    console.print(f"[+] [bold green]TorchScript 导出完成[/bold green]，文件已保存至: [cyan]{output_path}[/cyan]")

def main():
    config_file = Path("./config.yaml")
    if not config_file.exists():
        console.print(f"[bold red]错误：找不到配置文件 {config_file.absolute()}[/bold red]")
        return

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg_full = yaml.safe_load(f)
        
    if 'kielas_rm_export' not in cfg_full:
        console.print("[bold red]错误：配置文件中缺少 'kielas_rm_export' 模块。[/bold red]")
        return
        
    cfg = cfg_full['kielas_rm_export']
    
    weights_path = Path(cfg['weights'])
    output_dir = Path(cfg['output_dir'])
    formats = cfg.get('formats', [])
    input_size = cfg.get('input_size', [416, 416])
    
    # --- 修改点 2: 读取 reg_max 与 num_classes ---
    reg_max = cfg.get('reg_max', 16)
    num_classes = cfg.get('num_classes', 12)

    if not weights_path.exists():
        console.print(f"[bold red]错误：权重文件不存在 {weights_path.absolute()}[/bold red]")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[*] [bold cyan]正在初始化模型并加载权重...[/bold cyan]")
    device = torch.device('cpu') 
    
    # --- 修改点 3: 实例化时传入 reg_max 和 num_classes ---
    model = RMDetector(reg_max=reg_max, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    model.eval() 
    
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0], device=device)
    model_name = weights_path.stem
    
    if "onnx" in formats:
        onnx_path = output_dir / f"{model_name}.onnx"
        export_onnx(model, dummy_input, onnx_path, cfg)
        
    if "torchscript" in formats:
        ts_path = output_dir / f"{model_name}.pt"
        export_torchscript(model, dummy_input, ts_path)

    console.print("\n[bold green]所有导出任务执行完毕！[/bold green]")
    console.print(Panel(f"模型导出目录: [cyan]{output_dir.absolute()}[/cyan]", title="任务完成"))

if __name__ == "__main__":
    with torch.no_grad():
        main()