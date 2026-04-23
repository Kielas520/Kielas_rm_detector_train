# data_process/src/__init__.py

# 1. 导入各模块核心功能
from .purify import purify_dataset_pipeline
from .balance import balance_dataset_pipeline
from .split import split_dataset_pipeline
from .visiualize import visualize_dataset

# 2. 定义三个逻辑分分组（即你想要的“三个包”）
# 这样外部调用者可以通过 __all__ 知道哪些是核心接口

# 组一：预处理包 (Preprocessing)
PREPROCESS = [
    'purify_dataset_pipeline', 
    'balance_dataset_pipeline'
]

# 组二：流水线构建包 (Pipeline)
PIPELINE = [
    'split_dataset_pipeline', 
]

# 组三：可视化与分析包 (Analysis)
ANALYSIS = [
    'visualize_dataset'
]

# 3. 汇总到 __all__
# 只有出现在这个列表里的名字，在 from src import * 时才会被导入
__all__ = PREPROCESS + PIPELINE + ANALYSIS