import os
import cv2
import shutil
import bisect
import subprocess
import rosbag2_py
from concurrent.futures import ThreadPoolExecutor
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
import sys

def source_env(script_path):
    """加载 ROS2 接口环境并同步更新 Python 搜索路径"""
    script_path = os.path.expanduser(script_path)
    if not os.path.exists(script_path):
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
        self.bridge = CvBridge()
        # 文件夹与 ID 映射关系
        self.folder_map = {
            "hero_blue_data": 0,      # B1
            "infantry_blue_data": 2,  # B3
            "sentry_blue_data": 5,    # B7
            "hero_red_data": 6,       # R1
            "infantry_red_data": 8,   # R3
            "sentry_red_data": 11     # R7
        }
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.original_dir = os.path.join(self.root_dir, "extract_ros2_bag/original")
        self.raw_data_dir = os.path.join(self.root_dir, "data/raw")

    def prepare_directory(self, target_path):
        """检查并清理目录"""
        try:
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
                print(f"[清理] 已重置目录: {target_path}")
            os.makedirs(target_path, exist_ok=True)
        except Exception as e:
            print(f"[错误] 无法准备目录 {target_path}: {e}")

    def get_reader(self, path):
        storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        return reader

    def process_single_bag(self, folder_name, target_id):
        """处理单个 Bag 的核心逻辑"""
        bag_path = os.path.join(self.original_dir, folder_name)
        if not os.path.exists(bag_path):
            print(f"[跳过] 未找到目录: {bag_path}")
            return

        # 针对每个 ID 文件夹进行清理和重新创建
        # 路径结构为: data/raw/{target_id}/photos 和 data/raw/{target_id}/labels
        base_id_dir = os.path.join(self.raw_data_dir, str(target_id))
        self.prepare_directory(base_id_dir) # 这里会把 ID 下的 photos 和 labels 一起清空

        photo_dir = os.path.join(base_id_dir, "photos")
        label_dir = os.path.join(base_id_dir, "labels")
        os.makedirs(photo_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        print(f"[开始] 正在处理: {folder_name} (ID: {target_id})")

        try:
            # 第一步：预读取所有标签信息
            label_indices = []
            reader = self.get_reader(bag_path)
            while reader.has_next():
                (topic, data, t) = reader.read_next()
                if topic == '/detector/armors_debug_info':
                    # 确保 ArmorsDebugMsg 已经正确 import，否则此处会报错
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

            # 第二步：流式读取图像并匹配保存
            reader = self.get_reader(bag_path)
            img_count = 0
            while reader.has_next():
                (topic, data, t) = reader.read_next()
                if topic == '/image_raw':
                    idx = bisect.bisect_left(label_timestamps, t)
                    best_idx = idx
                    if idx > 0 and (idx == len(label_timestamps) or abs(t - label_timestamps[idx-1]) < abs(t - label_timestamps[idx])):
                        best_idx = idx - 1

                    # 匹配时间戳容差 (50ms)
                    if abs(t - label_timestamps[best_idx]) < 50_000_000:
                        msg = deserialize_message(data, Image)
                        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                        file_name = f"{img_count:06d}"
                        cv2.imwrite(os.path.join(photo_dir, f"{file_name}.jpg"), cv_img)

                        matched_armors = label_indices[best_idx][1]
                        with open(os.path.join(label_dir, f"{file_name}.txt"), 'w') as f:
                            for a in matched_armors:
                                pts_str = " ".join(map(str, a['pts']))
                                f.write(f"{a['id']} {a['color']} {pts_str}\n")

                        img_count += 1

            print(f"[完成] {folder_name}: 生成 {img_count} 组数据。")

        except Exception as e:
            print(f"[错误] 处理 {folder_name} 时发生异常: {e}")

    def extract(self):
        print(f"准备并发处理 {len(self.folder_map)} 个数据包...")
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(self.process_single_bag, folder_name, target_id)
                for folder_name, target_id in self.folder_map.items()
            ]
            for future in futures:
                future.result()

if __name__ == '__main__':
    source_path = "~/DT46_V/install/setup.bash"

    # 检查当前是否已经处于 source 后的环境中（通过检查自定义变量）
    # 假设你的 setup.bash 会设置 LD_LIBRARY_PATH 包含 DT46_V
    if "DT46_V" not in os.environ.get("LD_LIBRARY_PATH", ""):
        print("检测到环境未完全加载，正在重新注入环境变量并重启脚本...")

        # 加载环境
        source_env(source_path)

        # 强制更新 LD_LIBRARY_PATH 后重启自身
        # 这会开启一个全新的 Python 进程，它将拥有正确的 .so 加载路径
        os.execv(sys.executable, ['python3'] + sys.argv)

    # 如果代码运行到这里，说明已经是“第二次启动”，环境已经 OK 了
    from rm_interfaces.msg import ArmorsDebugMsg
    extractor = RosBagExtractor()
    extractor.extract()
