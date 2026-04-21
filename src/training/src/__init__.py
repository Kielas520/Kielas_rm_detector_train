# training/src/__init__.py

# 1. 导入核心组件
from .model import RMDetector, decode_tensor, keypoint_nms
from .loss import RMDetLoss
from .datasets import RMArmorDataset

# 2. 如果你想把训练和导出的入口函数也暴露出来
# 注意：这需要你在 train.py 和 export.py 中将核心逻辑封装进函数（见下文建议）
# from .train import main as run_training
# from .export import main as run_export

# 3. 定义逻辑分组
# 模型与结构
MODEL_COMPONENTS = [
    'RMDetector',
    'decode_tensor',
    'keypoint_nms'
]

# 训练逻辑
TRAIN_COMPONENTS = [
    'RMDetLoss',
    'RMArmorDataset'
]

# 4. 汇总导出
__all__ = MODEL_COMPONENTS + TRAIN_COMPONENTS