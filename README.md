# K-Vision

K-Vision 是一个针对 RoboMaster (RM) 视觉任务设计的高效、端到端关键点检测神经网络框架。项目涵盖了从 ROS2 Bag 数据提取、交互式困难负样本采集、自动化清洗与均衡、极端数据增强、模型训练，到最终 ONNX 导出和硬同步推理的完整工作流。

## 🎯 类别映射 (Class IDs)

系统当前动态支持 13 个类别（可根据 `config.yaml` 灵活增删），默认映射如下：

| 编号 | 含义             | 序号 (Class ID) |
|------|------------------|-----------------|
| B1   | 蓝方 1 号 装甲板   | 0               |
| B2   | 蓝方 2 号 装甲板   | 1               |
| B3   | 蓝方 3 号 装甲板   | 2               |
| B4   | 蓝方 4 号 装甲板   | 3               |
| B5   | 蓝方 5 号 装甲板   | 4               |
| B7   | 蓝方哨兵 装甲板    | 5               |
| R1   | 红方 1 号 装甲板   | 6               |
| R2   | 红方 2 号 装甲板   | 7               |
| R3   | 红方 3 号 装甲板   | 8               |
| R4   | 红方 4 号 装甲板   | 9               |
| R5   | 红方 5 号 装甲板   | 10              |
| R7   | 红方哨兵 装甲板    | 11              |
| Negative | 负样本 (灯管/反光带等) | 12              |

---

## 🚀 核心特性

* **统一的工作流终端**: 提供基于 `rich` 构建的 `main.py` 交互式控制台，一键调度数据预处理、模型训练、导出与推理演示。
* **高鲁棒性数据管线**:
    * **自动降维与兼容**: 自动兼容 10 维（含颜色/可见度）与 9 维（纯坐标）标签数据，流水线底层统一处理并补齐标志位。
    * **极端数据增强**: CPU 端进行几何变换与全天候背景融合；GPU 端极速计算高斯模糊、HSV 偏移、高增益噪声及 Bloom 光晕。
* **轻量级高精度架构**:
    * **网络结构**: Depthwise Separable Conv + SPPF (Backbone)，FPN + PAN (Neck)，搭配解耦头 (Decoupled Head) 直接回归 8 个角点坐标。
    * **复合损失函数**: 结合 Focal Loss (分类)、Distribution Focal Loss (DFL)、Wing Loss (回归)、OKS，以及独创的**装甲板中心结构化惩罚损失 (Structural Loss)**。
* **部署友好**: 深度适配 ONNX 与 TorchScript 导出，内置物理距离感知自定义 NMS，支持海康 (Hikvision) 工业相机原生调用与热更新参数。

---

## 📂 目录结构

```text
K-Vision/
├── config.yaml                # 全局统一配置文件 (数据、训练、导出、推理、采样)
├── main.py                    # 统一工作流终端控制台入口
├── pyproject.toml             # uv 项目依赖配置 (支持跨平台 Torch 安装)
├── ros2_bag/                  # ROS2 原始录制数据存放目录 (需手动创建)
├── src/                       # 核心源码目录
│   ├── data_process/          # 数据流水线: 异常清洗、类别均衡、划分、抽样可视化
│   ├── training/              # 训练模块: 网络构建、复合损失、数据增强、Dataloader
│   ├── demo/                  # 推理模块: Detector 封装、跨尺度解码、NMS、相机流接入
│   └── tools/                 # 实用工具链
│       ├── extract_ros2_bag.py# ROS2 rosbag 自动化解析与同步提取
│       ├── labels.py          # 交互式困难负样本提取工具 (视频抽帧打标)
│       ├── background.py      # 室内背景数据集自动下载与分辨率过滤
│       ├── scaler.py          # 像素尺寸辅助测量工具 (调参辅助)
│       └── get_env.py         # CUDA 环境检测脚本
```

---

## 🛠️ 环境配置

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 作为极速包管理器，支持 Python 3.11 环境。项目已配置跨平台依赖逻辑。

```bash
# 1. 安装 uv (如果尚未安装)
pip install uv

# 2. 同步并安装所有依赖 (Windows 下自动拉取 cu128 预览版 PyTorch)
uv sync

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

---

## 🚦 快速开始

所有的核心功能模块均已集成至统一的终端菜单中。激活虚拟环境后运行：

```bash
python main.py
```

在控制台中，输入对应的序号即可执行：
1. **数据预处理**：一键执行清洗 (`Purify`)、均衡 (`Balance`)、拆分 (`Split`) 与可视化 (`Visualize`)。
2. **开启模型训练**：基于 `config.yaml` 的超参数启动多进程混合增强训练。
3. **模型格式导出**：将 `.pth` 权重导出为部署级 `.onnx`。
4. **实时推理演示**：调用本地或工业相机进行目标检测与效果验证。

---

## ⚙️ 详细工作流指南

### 1. 数据准备 (Data Preparation)

#### A. ROS2 Bag 数据提取
将录制好的 ROS2 数据放入项目根目录的 `ros2_bag/` 文件夹下。
* **必需的话题 (Topics)**: `/detector/img_debug` (图像) 和 `/detector/armors_debug_info` (装甲板角点与标签)。
* **文件夹命名规范**: 提取脚本通过文件夹名称自动映射到对应的 Class ID，命名必须如下：
  * `hero_blue_data` (映射至 0) | `infantry_blue_data` (映射至 2) | `sentry_blue_data` (映射至 5)
  * `hero_red_data` (映射至 6) | `infantry_red_data` (映射至 8) | `sentry_red_data` (映射至 11)
* 运行 `python src/tools/extract_ros2_bag.py`，数据将自动解包并同步提取到 `data/raw` 中。

#### B. 困难负样本采集 (Hard Negatives - Class ID 12)
针对赛场中极易引起误识别的日光灯管、高反光边缘等，需录制视频并进行采样：
1. 在 `config.yaml` 的 `negative_sampler` 节点下配置视频路径 `video_path`。
2. 运行 `python src/tools/labels.py` 启动交互式采集界面。
3. 播放过程中：按 `D` 键跳过帧，按 `A` 键进入当前帧标定。
4. **标定操作**：通过键盘 `W/S` 或 `0-9` 将当前类别切换至 `12`。依次鼠标左键点击四个角点（右键撤销），按 `Enter` 键保存。数据会自动存入指定目录，并在后续随流水线混入训练集。

#### C. 背景图片准备
运行 `python src/tools/background.py` 自动下载 MIT indoorCVPR_09 数据集至本地，用于训练过程中的背景动态融合增强。

### 2. 数据处理与模型训练 (Pipeline & Training)

* **数据清洗与兼容**：在 `main.py` 中选择 `1` 执行全流程。脚本会自动剔除无效数据、按类别下采样（防过拟合），并对新老数据进行维度统一（抛弃冗余特征，自动补全标志位）。
* **启动训练**：在 `main.py` 中选择 `2`。
    * 支持动态类别读取，系统会根据 `config.yaml` 中配置的 `num_classes` 自动构建网络结构和多分类损失分支。
    * **半在线洗牌 (Shuffle Interval)**：每隔固定 Epoch 自动刷新多进程环境随机种子，极大提升数据增强泛化性。
    * **可视化特征钩子 (Hook)**：训练/验证结束时，会自动导出深层特征图 (P3, P4, P5) 及预测对比结果图，方便排查模型检测盲区。

### 3. 模型导出与工业相机推理 (Export & Inference)

* **一键导出**：在 `main.py` 中选择 `3` 导出模型。在已安装 `onnxsim` 的环境下，程序将自动执行常量折叠与计算图极致精简。
* **硬件对接**：
    * `src/demo/detector.py` 提供高层级检测器，内置基于关键点距离的高效跨尺度 NMS。
    * 原生支持海康 (Hikvision) 工业相机，在推理画面下按 `W/S` 键可实时调节相机的物理底层曝光度，应对复杂赛场打光。

---

## 📝 配置文件说明 (`config.yaml`)

所有的超参数调整集中于 `config.yaml`，主要包含以下核心节点：

* `negative_sampler`: 负样本采样器的视频源、导出路径与帧步长配置。
* `kielas_rm_train.dataset`: 包含数据流转路径、类别均衡上限 (`max_samples_per_class`) 以及各类极端的混合数据增强概率 (`bloom_prob`, `hsv_s_gain` 等)。
* `kielas_rm_train.train`: 训练核心配置，包括 `num_classes` (当前为 13)、白名单 `class_id`、学习率调度、综合评估权重 (`metric_weights`) 以及自动早停机制。
* `kielas_rm_export` / `kielas_rm_demo`: 指定导出时的输入尺寸、支持的类别数及相机硬件调用类型 ("usb" 或 "hik")。