# 数据增强
数据增强要根据图像数据的类型、任务需求和场景来选择数据增强的方式。
装甲板要保持相对完整性，可能要保持70~80%的完整度，对于颜色和结构的数据增强要谨慎。
YOLO 内置增强 ：在 model.train() 的 hyp 参数或 hyp.yaml 中设置，YOLO 还是很智能的。
但是可以补充其他处理方式，提高RM场景化。

## 可考虑加入visibility flag，但是不推荐大量加入，根据需求添加
完全现形（所有4个角点清晰可见，visibility=2）：
模糊现形（部分角点模糊/低对比/被轻微遮挡，visibility=1）：
完全不现形（完全没有角点，visibility=0）：

——————————————————————————————————————————————————————————————————————————————————————————————————————————————————

## 一些基于几何数据增强方式（Image Manipulation）
![alt text](DA_01.jpg)
旋转 Rotation/Degrees
平移 Translation
透视 Perspective
错切 shearing（有点透视加切除的意思，就是把图像由矩形变成平行四边形）

![alt text](DA_02.jpg)
Croppin & Resize （裁剪），在YOLO-Pose 中，是Scale（缩放）
Noise Injection（注入噪声）

fliping 没有必要，因为不会出现镜像 “专家板”
对于颜色的处理要谨慎，可能会导致装甲板识别失效

### 图像擦除
![alt text](DA_03.jpg)
Random erasing 是一种像剪切一样随机擦除图像中子区域

### 遮挡
GridMask 基于网格掩码的数据增强方式
![alt text](DA_04.jpg)

### 混合样本
![alt text](DA_05.jpg)

### 局部降低分辨率
CutBlur
![alt text](DA_06.jpg)


——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
## 一些基于赛场的推荐
### 运动模糊
Motion Blur
应对比赛中装甲板高速移动或相机抖动常见

### 局部对比度增强
Random Brightness/Contrast + CLAHE
模拟不同灯光条件（强光、阴影）。对装甲板灯条亮度变化很有帮助

### Mosaic + Close Mosaic（Ultralytics 内置）
保留 mosaic: 1.0，但设置 close_mosaic: 10~15（最后几个 epoch 关闭），能自然引入多目标、小目标、部分现形，同时后期让模型稳定学习完整样本。

### Albumentations 高级组合（最推荐的落地方式）
强烈建议把 Ultralytics 原生增强 + Albumentations 自定义 pipeline 结合使用（Ultralytics 会自动检测并使用）。
这样可以精细控制每个变换对 keypoints 的同步，并加入 visibility 处理逻辑。

### 复制-粘贴增强
适应场景
Copy-Paste Augmentation

知乎上非常全的数据增强介绍： https://zhuanlan.zhihu.com/p/598985864