import tarfile
import requests
import shutil
import random
import io
import yaml  # 需要安装: pip install pyyaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
from PIL import Image  # 需要安装: pip install pillow
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.prompt import Confirm  # 用于交互确认

console = Console()

@dataclass
class DownloadConfig:
    """下载与提取任务的全局配置"""
    url: str = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
    limit: int = 5000
    base_path: str = "./background"
    tar_file_name: str = "indoorCVPR_09.tar"
    flag_file_name: str = "done.flag"
    force_refresh: bool = False
    
    max_resolution: int = 1280
    min_resolution: int = 320
    use_proxy: bool = False
    proxies: Optional[Dict[str, str]] = field(default_factory=lambda: {
        "http": "http://127.0.0.1:7897",
        "https": "http://127.0.0.1:7897",
    })

def sync_with_yaml(config: DownloadConfig, yaml_path: str = "config.yaml"):
    """从 YAML 中读取配置并覆盖默认参数"""
    try:
        if not Path(yaml_path).exists():
            return
            
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            root_cfg = data.get('kielas_rm_train', {}).get('tools', {})
            
            # 1. 读取下载专用配置
            bg_info = root_cfg.get('background', {})
            if bg_info:
                config.url = bg_info.get('url', config.url)
                config.tar_file_name = bg_info.get('tar_name', config.tar_file_name)
                config.limit = bg_info.get('limit', config.limit)
                config.max_resolution = bg_info.get('max_res', config.max_resolution)
                config.min_resolution = bg_info.get('min_res', config.min_resolution)
                config.use_proxy = bg_info.get('use_proxy', config.use_proxy)
                config.proxies = bg_info.get('proxies', config.proxies)

            # 2. 读取路径配置
            aug_cfg = root_cfg.get('augment', {})
            config.base_path = aug_cfg.get('bg_dir', config.base_path)
            
            console.print(f"[green]成功从 {yaml_path} 加载配置[/green]")
    except Exception as e:
        console.print(f"[yellow]加载 {yaml_path} 失败，使用内置默认值。错误: {e}[/yellow]")

def download_and_extract(config: DownloadConfig):
    root = Path(config.base_path).resolve()
    flag_file = root / config.flag_file_name
    tar_path = root / config.tar_file_name
    
    # 检测是否已经处理完成
    if flag_file.exists() and not config.force_refresh:
        should_reprocess = Confirm.ask(
            f"[yellow]检测到数据已在 {config.base_path} 处理完成，是否删除并重新处理？[/yellow]",
            default=False
        )
        if should_reprocess:
            config.force_refresh = True
        else:
            console.print("[green]跳过处理。[/green]")
            return

    # 执行清理逻辑
    if config.force_refresh:
        console.print(Panel(f"[bold red]强制刷新：正在清理目录 {root}[/bold red]"))
        if root.exists():
            for item in root.iterdir():
                try:
                    if item.is_dir(): shutil.rmtree(item)
                    else: item.unlink()
                except Exception as e:
                    console.print(f"[red]清理失败 {item.name}: {e}[/red]")
    
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    # 下载逻辑
    if not flag_file.exists():
        headers = {}
        resume_byte = tar_path.stat().st_size if tar_path.exists() else 0
        if resume_byte > 0:
            headers['Range'] = f'bytes={resume_byte}-'
            console.print(f"[yellow]续传模式：从 {resume_byte / 1024**2:.2f} MB 开始...[/yellow]")

        try:
            if config.use_proxy:
                response = requests.get(config.url, headers=headers, stream=True, timeout=30, proxies=config.proxies)
            else:
                response = requests.get(config.url, headers=headers, stream=True, timeout=30)
            # 如果服务器不支持 Range 或文件已改变，重置下载
            if response.status_code == 200:
                resume_byte = 0
                mode = 'wb'
            elif response.status_code == 206:
                mode = 'ab'
            elif response.status_code == 416: # Range 不符合要求
                resume_byte = 0
                mode = 'wb'
                response = requests.get(config.url, stream=True, timeout=30, proxies=config.proxies)
            else:
                console.print(f"[bold red]下载失败，状态码: {response.status_code}[/bold red]")
                return

            total_size = int(response.headers.get('content-length', 0)) + resume_byte
            
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task_id = progress.add_task("下载数据源", total=total_size, completed=resume_byte)
                with tar_path.open(mode) as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))
        except Exception as e:
            console.print(f"[bold red]网络异常:[/bold red] {e}")
            return

    # 提取与过滤逻辑
    try:
        if flag_file.exists():
            console.print("[green]检测到标记文件，跳过提取。[/green]")
            return

        with tarfile.open(tar_path, 'r:*') as tar:
            console.print("[blue]执行分辨率过滤采样中...[/blue]")
            # 过滤出图片文件
            all_members = [
                m for m in tar.getmembers() 
                if m.isfile() and m.name.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            random.shuffle(all_members)
            extracted_count = 0
            
            with Progress(console=console) as progress:
                task = progress.add_task("[cyan]过滤并保存图片...", total=config.limit)
                for member in all_members:
                    if extracted_count >= config.limit: 
                        break
                    
                    f_obj = tar.extractfile(member)
                    if f_obj is None: 
                        continue
                        
                    try:
                        img_data = f_obj.read()
                        with Image.open(io.BytesIO(img_data)) as img:
                            w, h = img.size
                            # 分辨率过滤
                            if max(w, h) > config.max_resolution or min(w, h) < config.min_resolution:
                                continue
                            
                            # 保存到根目录，去除原有的压缩包内路径
                            target_path = root / Path(member.name).name
                            img.save(target_path)
                            extracted_count += 1
                            progress.update(task, advance=1)
                    except Exception:
                        continue
            
        flag_file.touch()
        if tar_path.exists(): 
            tar_path.unlink()
            
        console.print(Panel(f"[bold green]任务完成！[/bold green]\n共计: {extracted_count} 张\n路径: {root}"))
        
    except Exception as e:
        console.print(f"[bold red]处理出错:[/bold red] {e}")

if __name__ == "__main__":
    # 初始化默认配置
    cfg = DownloadConfig()
    
    # 自动从 yaml 覆盖配置
    sync_with_yaml(cfg, "config.yaml")
    
    # 执行下载与提取
    download_and_extract(cfg)