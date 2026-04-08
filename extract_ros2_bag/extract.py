import os
import cv2
import shutil
import bisect
import subprocess
import rosbag2_py
from concurrent.futures import ProcessPoolExecutor
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import sys
from pathlib import Path

def source_env(script_path):
    """加载 ROS2 接口环境并同步更新 Python 搜索路径"""
    script_path = Path(script_path).expanduser()
    if not script_path.exists():
        print(f"[跳过] 未找到环境脚本: {script_path}")
        return

    try:
        # 执行 source 并获取所有环境变量
        command = f"bash -c 'source {script_path} && env'"
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, text=True)

        for line in proc.stdout:
            if "=" in line:
                key, _, value = line.partition("=")
                value = value.strip()
                os.environ[key] = value

                # 关键步骤：如果发现 PYTHONPATH，将其中的路径加入到当前 Python 的搜索路径中
                if key == "PYTHONPATH":
                    for path in value.split(":"):
                        if path and path not in sys.path:
                            sys.path.insert(0, path) # 插入到最前面，确保优先找到

        proc.communicate()
        print(f"[环境] 成功 Source 且已更新 sys.path")
    except Exception as e:
        print(f"[环境] 加载环境失败: {e}")

class RosBagExtractor:
    def __init__(self):
        # 文件夹与 ID 映射关系
        self.folder_map = {
            "hero_blue_data": 0,      # B1
            "infantry_blue_data": 2,  # B3
            "sentry_blue_data": 5,    # B7
            "hero_red_data": 6,       # R1
            "infantry_red_data": 8,   # R3
            "sentry_red_data": 11     # R7
        }
        # 使用 Path API 获取根目录和相关路径
        self.root_dir = Path(__file__).resolve().parent.parent
        self.original_dir = self.root_dir / "extract_ros2_bag" / "original"
        self.raw_data_dir = self.root_dir / "data" / "raw"

    def prepare_directory(self, target_path: Path):
        """检查并清理目录"""
        try:
            if target_path.exists():
                shutil.rmtree(target_path)
                print(f"[清理] 已重置目录: {target_path}")
            target_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[错误] 无法准备目录 {target_path}: {e}")

    def get_reader(self, path: Path):
        storage_options = rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        return reader

    def process_single_bag(self, folder_name, target_id):
        """处理单个 Bag 的核心逻辑"""
        # 在 worker 内部导入自定义消息，确保在多进程中环境已经就绪
        from rm_interfaces.msg import ArmorsDebugMsg
        
        bag_path = self.original_dir / folder_name
        if not bag_path.exists():
            print(f"[跳过] 未找到目录: {bag_path}")
            return

        base_id_dir = self.raw_data_dir / str(target_id)
        self.prepare_directory(base_id_dir)

        photo_dir = base_id_dir / "photos"
        label_dir = base_id_dir / "labels"
        photo_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        print(f"[开始] 正在处理: {folder_name} (ID: {target_id})")
        
        # 在进程内部实例化，避免跨进程传递 C++ 对象的序列化问题
        bridge = CvBridge()

        try:
            # ================= 第一步：极速预读取所有标签信息 =================
            label_indices = []
            reader = self.get_reader(bag_path)
            
            # 使用过滤器，底层直接过滤掉海量的图片数据，极大提升第一步的速度
            storage_filter = rosbag2_py.StorageFilter(topics=['/detector/armors_debug_info'])
            reader.set_filter(storage_filter)
            
            while reader.has_next():
                (topic, data, t) = reader.read_next()
                msg = deserialize_message(data, ArmorsDebugMsg)
                armor_list = []
                for a in msg.armors_debug:
                    armor_list.append({
                        'id': a.armor_id, 'color': a.color,
                        'pts': [a.l_light_up_dx, a.l_light_up_dy, a.l_light_down_dx, a.l_light_down_dy,
                                a.r_light_up_dx, a.r_light_up_dy, a.r_light_down_dx, a.r_light_down_dy]
                    })
                label_indices.append((t, armor_list))

            if not label_indices:
                print(f"[警告] {folder_name} 中未找到标签数据。")
                return

            label_indices.sort(key=lambda x: x[0])
            label_timestamps = [x[0] for x in label_indices]

            # ================= 第二步：流式读取图像并匹配保存 =================
            reader = self.get_reader(bag_path)
            
            # 同样为第二步加上图片过滤器，跳过其它无关 topic
            image_filter = rosbag2_py.StorageFilter(topics=['/image_raw'])
            reader.set_filter(image_filter)
            
            img_count = 0
            while reader.has_next():
                (topic, data, t) = reader.read_next()
                
                idx = bisect.bisect_left(label_timestamps, t)
                best_idx = idx
                if idx > 0 and (idx == len(label_timestamps) or abs(t - label_timestamps[idx-1]) < abs(t - label_timestamps[idx])):
                    best_idx = idx - 1

                # 匹配时间戳容差 (50ms)
                if abs(t - label_timestamps[best_idx]) < 50_000_000:
                    msg = deserialize_message(data, Image)
                    cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")

                    file_name = f"{img_count:06d}"

                    cv2.imwrite(str(photo_dir / f"{file_name}.jpg"), cv_img)

                    matched_armors = label_indices[best_idx][1]

                    with open(label_dir / f"{file_name}.txt", 'w') as f:
                        for a in matched_armors:
                            pts_str = " ".join(map(str, a['pts']))
                            f.write(f"{a['id']} {a['color']} {pts_str}\n")

                    img_count += 1

            print(f"[完成] {folder_name}: 生成 {img_count} 组数据。")

        except Exception as e:
            print(f"[错误] 处理 {folder_name} 时发生异常: {e}")

    def extract(self):
        print(f"准备并发处理 {len(self.folder_map)} 个数据包 (使用多进程)...")
        # 使用多进程替代多线程，绕过全局解释器锁 (GIL)
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(self.process_single_bag, folder_name, target_id)
                for folder_name, target_id in self.folder_map.items()
            ]
            for future in futures:
                future.result()

if __name__ == '__main__':
    source_path = "~/DT46_V/install/setup.bash"

    if "DT46_V" not in os.environ.get("LD_LIBRARY_PATH", ""):
        print("检测到环境未完全加载，正在重新注入环境变量并重启脚本...")
        source_env(source_path)
        os.execv(sys.executable, ['python3'] + sys.argv)

    extractor = RosBagExtractor()
    extractor.extract()