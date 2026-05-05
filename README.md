**实验环境与参数**

实验采用 Ubuntu 22.04 系统，硬件为 NVIDIA RTX 4090 (48GB)，软件环境配置 Python 3.12、PyTorch 2.7.0、CUDA 12.8，基于 Ultralytics YOLO11 框架开展训练。训练参数设置如下：输入图像尺寸 200×200，批次大小 16，最大训练轮数 200，采用 SGD 优化器，初始学习率 0.01，动量 0.937，权重衰减 0.0005，并使用 Mosaic、随机翻转、随机擦除等数据增强策略，开启混合精度与数据缓存加速训练。

**实验结果**
1. NEU-DET 数据集上不同模块的消融研究（M1：RGA；M2：MDLA；M3：BiFN）
   
1.1. baseline:下采样功能用nn.AvgPool2d 替代（无卷积、纯池化），模型为yaml/baseline.yaml，训练结果得到runs/detect/baseline

<img width="1188" height="353" alt="image" src="https://github.com/user-attachments/assets/51d55b88-fd58-4119-9d4a-ea5ba45a983f" />




1.2. baseline+M1:对1.1中模型训练得到的结果Recall<0.5进行RGA重构，小于0.5的只有crazing，模型同1.1，训练结果得到runs/detect/baseline+M1
<img width="1183" height="333" alt="image" src="https://github.com/user-attachments/assets/22f6f886-e363-477c-88c3-5f36b57b489f" />


1.3. baseline+M1+M2:MDLA多膨胀局部注意力进行模型缝合，在ultralytics/nn/modules/block.py中添加class MDLA,然后在同级文件夹中的_init_.py和上一级文件夹中的task.py进行注册，模型为yaml/baseline+M1+M2.yaml，训练结果得到runs/detect/baseline+M1+M2,代码如下：
```python
class MDLA(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        assert c1 == c2
        self.c = c1
        self.heads = 4  # 固定头数
        self.dilation_list = [1, 2, 3, 4]
        
        # 1x1 卷积
        self.qkv = nn.Conv2d(self.c, self.c * 3, 1, bias=False)
        self.proj = nn.Conv2d(self.c, self.c, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).chunk(3, 1)  # [B, 3C, H, W]
        q, k, v = map(lambda t: t.view(B, self.heads, C//self.heads, H*W), qkv)
        out = 0
        for d in self.dilation_list:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / (q.size(-1) ** 0.5))
            attn = F.softmax(attn, dim=-1)
            out = out + (attn @ v)
        out = out.view(B, C, H, W)
        out = self.proj(out)
        return x + out
```
<img width="1186" height="352" alt="image" src="https://github.com/user-attachments/assets/b87aa989-6696-4799-bd0e-2c0216ca5a4c" />

1.4. baseline+M1+M2+M3(conv):模型为yaml/baseline+M1+M2+M3(conv).yaml，训练结果得到runs/detect/baseline+M1+M2+M3(conv)

<img width="1182" height="356" alt="image" src="https://github.com/user-attachments/assets/591a6f8a-dcb4-4484-a6b6-a49e72eba00d" />

1.5. baseline+M1+M2+M3(scdown):SCDown下采样,与1.3相同在block.py中修改class SCDown的代码,模型为yaml/baseline+M1+M2+M3(scdown).yaml，训练结果得到runs/detect/baseline+M1+M2+M3(scdown),代码如下：
```python
class SCDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        # 1x1 卷积（通道变换）
        self.pw = Conv(c1, c2, 1, 1)
        # 3x3 卷积（空间下采样，不使用 groups）
        self.dw = Conv(c2, c2, k, s, autopad(k))

    def forward(self, x):
        return self.dw(self.pw(x))
```
<img width="1171" height="367" alt="image" src="https://github.com/user-attachments/assets/0f7025f9-3cf6-4937-989a-e3e58b343533" />

1.6. 总结：RGA标签重构可显著提升YOLO11的检测精度，在此基础上引入MDLA模块能进一步提升mAP50，而BiFN却导致实验结果的下降。
<img width="850" height="169" alt="image" src="https://github.com/user-attachments/assets/bb111623-ece3-4630-aaea-6ba3e82a5c20" />


<img width="413" height="151" alt="image" src="https://github.com/user-attachments/assets/dba6b27a-380a-4f74-8000-c3379ef3772b" />

