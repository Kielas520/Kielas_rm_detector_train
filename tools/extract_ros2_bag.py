import os
import cv2
import shutil
import subprocess
import rosbag2_py
from concurrent.futures import ProcessPoolExecutor
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import sys
from pathlib import Path
import multiprocessing
import threading

from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn
)
from rich.console import Console

console = Console()

def source_env(script_path):
    """加载 ROS2 环境"""
    script_path = Path(script_path).expanduser()
    if not script_path.exists(): return
    try:
        command = f"bash -c 'source {script_path} && env'"
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, text=True)
        for line in proc.stdout:
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key] = value.strip()
                if key == "PYTHONPATH":
                    for path in value.strip().split(":"):
                        if path and path not in sys.path: sys.path.insert(0, path)
        proc.communicate()
    except: pass

class RosBagExtractor:
    def __init__(self):
        # 对应关系：文件夹名 -> 机器人ID
        self.folder_map = {
            "hero_blue_data": 0, "infantry_blue_data": 2, "sentry_blue_data": 5,
            "hero_red_data": 6, "infantry_red_data": 8, "sentry_red_data": 11
        }
        self.root_dir = Path(__file__).resolve().parent.parent
        self.original_dir = self.root_dir / "ros2_bag"
        self.raw_data_dir = self.root_dir / "data" / "raw"

    @staticmethod
    def process_single_bag(folder_name, target_id, progress_queue, task_id, original_dir, raw_data_dir):
        """带有刷新逻辑和错误统计的提取逻辑"""
        # 屏蔽 ROS2 底层冗余日志
        os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = ''
        os.environ['RCL_LOG_LEVEL'] = '40' 
        
        # 延迟导入自定义消息，避免多进程初始化冲突
        from rm_interfaces.msg import ArmorsDebugMsg
        
        stats = {"empty_pts": 0, "invalid_pts": 0, "match_fail": 0, "valid_frames": 0}
        
        bag_path = original_dir / folder_name
        if not bag_path.exists():
            progress_queue.put({"task_id": task_id, "type": "description", "value": "[red]目录缺失", "stats": stats})
            progress_queue.put({"task_id": task_id, "type": "done"})
            return

        # --- 刷新逻辑：如果旧的 raw 存在则删除 ---
        base_id_dir = raw_data_dir / str(target_id)
        try:
            if base_id_dir.exists():
                progress_queue.put({"task_id": task_id, "type": "description", "value": "[yellow]清理旧数据...", "stats": stats})
                shutil.rmtree(base_id_dir)
            
            # 重新创建目录结构
            photo_dir = base_id_dir / "photos"
            label_dir = base_id_dir / "labels"
            photo_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            progress_queue.put({"task_id": task_id, "type": "description", "value": f"[red]刷新失败: {str(e)[:10]}", "stats": stats})
            progress_queue.put({"task_id": task_id, "type": "done"})
            return

        bridge = CvBridge()
        storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

        try:
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
            
            metadata = reader.get_metadata()
            total_images = next((t.message_count for t in metadata.topics_with_message_count if t.topic_metadata.name == '/detector/img_debug'), 0)
            progress_queue.put({"task_id": task_id, "type": "total", "value": total_images})
            progress_queue.put({"task_id": task_id, "type": "description", "value": "[cyan]正在同步...", "stats": stats})

            # 1. 读取并索引标签 (Topic: /detector/armors_debug_info)
            label_map = {}
            reader.set_filter(rosbag2_py.StorageFilter(topics=['/detector/armors_debug_info']))
            
            while reader.has_next():
                (_, data, _) = reader.read_next()
                msg = deserialize_message(data, ArmorsDebugMsg)
                t_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                
                armors = []
                for a in msg.armors_debug:
                    pts = [a.l_light_up_dx, a.l_light_up_dy,
                           a.l_light_down_dx, a.l_light_down_dy, 
                           a.r_light_up_dx, a.r_light_up_dy, a.r_light_down_dx, a.r_light_down_dy]
                    
                    # 简单过滤空标签
                    if all(p == 0 for p in pts[:4]):
                        stats["empty_pts"] += 1
                        continue
                    
                    armors.append({'id': a.armor_id, 'color': a.color, 'pts': pts})
                
                if armors:
                    label_map[t_ns] = armors

            # 2. 导出图像并进行时间戳匹配 (Topic: /detector/img_debug)
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
            reader.set_filter(rosbag2_py.StorageFilter(topics=['/detector/img_debug']))
            
            img_count, batch = 0, 0
            while reader.has_next():
                (_, data, _) = reader.read_next()
                msg = deserialize_message(data, Image)
                img_t = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                
                if img_t in label_map:
                    cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
                    name = f"{img_count:06d}"
                    cv2.imwrite(str(photo_dir / f"{name}.jpg"), cv_img)
                    
                    with open(label_dir / f"{name}.txt", 'w') as f:
                        for a in label_map[img_t]:
                            pts_str = ' '.join(map(str, a['pts']))
                            f.write(f"{a['id']} {a['color']} {pts_str}\n")
                    img_count += 1
                    stats["valid_frames"] = img_count
                else:
                    stats["match_fail"] += 1
                
                batch += 1
                if batch >= 20: # 每20帧更新一次UI，减少进程间通信开销
                    progress_queue.put({"task_id": task_id, "type": "advance", "value": batch, "stats": stats})
                    batch = 0

            progress_queue.put({"task_id": task_id, "type": "advance", "value": batch, "stats": stats})
            progress_queue.put({"task_id": task_id, "type": "description", "value": f"[green]提取完成", "stats": stats})
            progress_queue.put({"task_id": task_id, "type": "done"})
            
        except Exception as e:
            # 临时增加打印完整异常，方便定位
            print(f"\n[Error Details] {repr(e)}") 
            progress_queue.put({"task_id": task_id, "type": "description", "value": f"[red]异常: {str(e)[:15]}", "stats": stats})
            progress_queue.put({"task_id": task_id, "type": "done"})

    def extract(self):
        console.print(f"[bold green]ROS2 数据同步提取器 (自动刷新模式)[/bold green]")
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[name]}"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=25),
            TaskProgressColumn(),
            TextColumn("[red]空标:{task.fields[empty]}"),
            TextColumn("[yellow]丢弃:{task.fields[fail]}"),
            TimeRemainingColumn(),
            refresh_per_second=10
        )

        def update_ui():
            while True:
                msg = progress_queue.get()
                if msg is None: break
                tid = msg['task_id']
                
                if 'stats' in msg:
                    progress.update(tid, empty=msg['stats']['empty_pts'], fail=msg['stats']['match_fail'])
                
                if msg['type'] == 'total': progress.update(tid, total=msg['value'])
                elif msg['type'] == 'advance': progress.advance(tid, advance=msg['value'])
                elif msg['type'] == 'description': progress.update(tid, description=msg['value'])
                elif msg['type'] == 'done': progress.update(tid, completed=progress.tasks[tid].total or 1)

        with progress:
            task_ids = {}
            for name in self.folder_map.keys():
                tid = progress.add_task("就绪", name=f"{name:<18}", total=None, empty=0, fail=0)
                task_ids[name] = tid

            ui_thread = threading.Thread(target=update_ui, daemon=True)
            ui_thread.start()

            # 使用多进程并行处理多个 Bag
            with ProcessPoolExecutor(max_workers=min(len(self.folder_map), multiprocessing.cpu_count())) as executor:
                futures = [
                    executor.submit(
                        self.process_single_bag, 
                        n, 
                        self.folder_map[n], 
                        progress_queue, 
                        task_ids[n], 
                        self.original_dir, 
                        self.raw_data_dir
                    ) for n in self.folder_map.keys()
                ]
                for f in futures: f.result()

            progress_queue.put(None)
            ui_thread.join()

if __name__ == '__main__':
    source_path = "~/DT46_V/install/setup.bash"

    # 环境检查与重载
    if "DT46_V" not in os.environ.get("LD_LIBRARY_PATH", ""):
        source_env(source_path)
        os.execv(sys.executable, ['python3'] + sys.argv)

    RosBagExtractor().extract()