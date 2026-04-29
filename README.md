**实验环境与参数**

实验采用 Ubuntu 22.04 系统，硬件为 NVIDIA RTX 4090 (48GB)，软件环境配置 Python 3.12、PyTorch 2.7.0、CUDA 12.8，基于 Ultralytics YOLO11 框架开展训练。训练参数设置如下：输入图像尺寸 200×200，批次大小 16，最大训练轮数 200，采用 SGD 优化器，初始学习率 0.01，动量 0.937，权重衰减 0.0005，并使用 Mosaic、随机翻转、随机擦除等数据增强策略，开启混合精度与数据缓存加速训练。
NEU-DET 数据集上不同模块的消融研究（M1：RGA；M2：MDLA；M3：BiFN）					
Baseline	M1	M2	M3	mAP50	mAP50-95
√				0.746	0.437
√	√			0.855	0.566
√	√	√		0.814	0.502
√	√	√	√	0.814	0.502
<img width="637" height="151" alt="image" src="https://github.com/user-attachments/assets/9ddc34da-6098-4592-8aee-81362f547514" />
