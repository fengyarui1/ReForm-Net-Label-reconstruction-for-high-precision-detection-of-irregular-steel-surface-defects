# 一、对数据集NEU-DET的实验

## **实验环境与参数**

实验采用 Ubuntu 22.04 系统，硬件为 NVIDIA RTX 4090 (48GB)，软件环境配置 Python 3.12、PyTorch 2.7.0、CUDA 12.8，基于 Ultralytics YOLO11 框架开展训练。训练参数设置如下：输入图像尺寸 200×200，批次大小 16，最大训练轮数 200，采用 SGD 优化器，初始学习率 0.01，动量 0.937，权重衰减 0.0005，并使用 Mosaic、随机翻转、随机擦除等数据增强策略，开启混合精度与数据缓存加速训练。

## **实验结果**

### 1. NEU-DET 数据集上不同模块的消融研究（M1：RGA；M2：MDLA；M3：BiFN）

#### 1.1. baseline

下采样功能用nn.AvgPool2d 替代（无卷积、纯池化），模型为yaml/baseline.yaml，训练结果得到runs/detect/baseline

<img width="1188" height="353" alt="image" src="https://github.com/user-attachments/assets/51d55b88-fd58-4119-9d4a-ea5ba45a983f" />

#### 1.2. baseline+M1

对1.1中模型训练得到的结果Recall<0.5进行RGA重构（2x2，以下的训练一致），小于0.5的只有crazing，模型同1.1，训练结果得到runs/detect/baseline+M1
<img width="1183" height="333" alt="image" src="https://github.com/user-attachments/assets/22f6f886-e363-477c-88c3-5f36b57b489f" />


#### 1.3. baseline+M1+M2

MDLA多膨胀局部注意力进行模型缝合，在ultralytics/nn/modules/block.py中添加class MDLA,然后在同级文件夹中的_init_.py和上一级文件夹中的task.py进行注册，模型为yaml/baseline+M1+M2.yaml，训练结果得到runs/detect/baseline+M1+M2,代码如下：
```python
class MDLA(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        assert c1 == c2    
        self.c = c1
        self.heads = 4  
        self.dilation_list = [1, 2, 3, 4]  
        # 1x1 卷积
        self.qkv = nn.Conv2d(self.c, self.c * 3, kernel_size=1, bias=False)
        # 输出投影卷积
        self.proj = nn.Conv2d(self.c, self.c, kernel_size=1, bias=False)
    def forward(self, x):
        B, C, H, W = x.shape    
        # 生成 Q, K, V: [B, 3*C, H, W] -> 拆分3个 [B, C, H, W]
        q, k, v = self.qkv(x).chunk(3, dim=1) 
        # 维度变换: 拆分多头 [B, heads, C//heads, H*W]
        q = q.view(B, self.heads, C // self.heads, H * W)
        k = k.view(B, self.heads, C // self.heads, H * W)
        v = v.view(B, self.heads, C // self.heads, H * W)
        # 初始化输出
        out = 0
        # 多空洞率注意力融合
        for _ in self.dilation_list:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / (q.size(-1) ** 0.5))
            attn = F.softmax(attn, dim=-1)
            out = out + (attn @ v)
        out = out.view(B, C, H, W)
        out = self.proj(out)
        
        return x + out
```
<img width="1186" height="352" alt="image" src="https://github.com/user-attachments/assets/b87aa989-6696-4799-bd0e-2c0216ca5a4c" />

#### 1.4. baseline+M1+M2+M3(conv)

模型为yaml/baseline+M1+M2+M3(conv).yaml，训练结果得到runs/detect/baseline+M1+M2+M3(conv)

<img width="1182" height="356" alt="image" src="https://github.com/user-attachments/assets/591a6f8a-dcb4-4484-a6b6-a49e72eba00d" />

#### 1.5. baseline+M1+M2+M3(scdown)

SCDown下采样,与1.3相同在block.py中修改class SCDown的代码,模型为yaml/baseline+M1+M2+M3(scdown).yaml，训练结果得到runs/detect/baseline+M1+M2+M3(scdown),代码如下：
```python
class SCDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        # 1x1 卷积（通道变换）
        self.pw = Conv(c1, c2, 1, 1)
        # 3x3 卷积（空间下采样）
        self.dw = Conv(c2, c2, k, s, autopad(k))

    def forward(self, x):
        return self.dw(self.pw(x))
```
<img width="1171" height="367" alt="image" src="https://github.com/user-attachments/assets/0f7025f9-3cf6-4937-989a-e3e58b343533" />

#### 1.6. 总结

RGA标签重构可显著提升YOLO11的检测精度，在此基础上引入MDLA模块能进一步提升mAP50，而BiFN却导致实验结果的下降。

<img width="850" height="193" alt="image" src="https://github.com/user-attachments/assets/0ae49d9f-1b5d-44df-9eb9-6d3d5cbad28a" />


### 2. NEU-DET 数据集上的网格实验

RGA标签重构代码如下，改变标签类别只需修改BOTTLENECK_CLASSES的值，值从0开始以此代表以下数组中的元素：['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']；改变重构的维数只需要修改GRID_SIZE的值。
```python
import os
import numpy as np

# ===================== 配置 =====================
BOTTLENECK_CLASSES = {0,4}  # 瓶颈类别
GRID_SIZE =  2 # 2×2网格划分

# 原始标签路径
INPUT_TRAIN = "./NEU-DET/labels/train_original"
INPUT_VAL   = "./NEU-DET/labels/val_original"
INPUT_TEST  = "./NEU-DET/labels/test_original"

# 重构后输出路径
OUTPUT_TRAIN = "./NEU-DET/labels/train"
OUTPUT_VAL   = "./NEU-DET/labels/val"
OUTPUT_TEST  = "./NEU-DET/labels/test"

# 自动创建文件夹
os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_VAL, exist_ok=True)
os.makedirs(OUTPUT_TEST, exist_ok=True)

# ===================== RGA 整图网格划分重构函数 =====================
def rga_reform_grid(txt_path, save_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    # 1. 先读取所有标签
    bboxes = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cls, x, y, w, h = map(float, line.split())
        cls = int(cls)
        bboxes.append((cls, x, y, w, h))

    # 2. 生成n×n网格（按整张图片划分）
    grid_w = 1.0 / GRID_SIZE
    grid_h = 1.0 / GRID_SIZE

    # 遍历所有网格
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # 计算网格的坐标和尺寸（YOLO格式：x_center, y_center, w, h）
            gx = grid_w * (j + 0.5)
            gy = grid_h * (i + 0.5)
            gw = grid_w
            gh = grid_h

            # 3. 检查这个网格是否和任何瓶颈类缺陷有交集，并记录重叠的类别ID
            overlap_cls = None
            for (cls, x, y, w, h) in bboxes:
                if cls not in BOTTLENECK_CLASSES:
                    continue

                # 计算网格和缺陷框的IOU（交集判断）
                # 网格坐标
                gx1, gy1 = gx - gw/2, gy - gh/2
                gx2, gy2 = gx + gw/2, gy + gh/2
                # 缺陷框坐标
                bx1, by1 = x - w/2, y - h/2
                bx2, by2 = x + w/2, y + h/2

                # 计算交集
                ix1 = max(gx1, bx1)
                iy1 = max(gy1, by1)
                ix2 = min(gx2, bx2)
                iy2 = min(gy2, by2)

                if ix1 < ix2 and iy1 < iy2:
                    overlap_cls = cls  # 保存实际的类别ID
                    break

            # 4. 如果有交集，就给这个网格打上原类别标签
            if overlap_cls is not None:
                new_lines.append(f"{overlap_cls} {gx:.6f} {gy:.6f} {gw:.6f} {gh:.6f}")

    # 5. 保留非瓶颈类的原始标签（完全不变）
    for (cls, x, y, w, h) in bboxes:
        if cls not in BOTTLENECK_CLASSES:
            new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))

# ===================== 批量处理三类数据集 =====================
def process_folder(input_dir, output_dir):
    for fname in os.listdir(input_dir):
        if fname.endswith(".txt"):
            rga_reform_grid(
                os.path.join(input_dir, fname),
                os.path.join(output_dir, fname)
            )

process_folder(INPUT_TRAIN, OUTPUT_TRAIN)
process_folder(INPUT_VAL, OUTPUT_VAL)
process_folder(INPUT_TEST, OUTPUT_TEST)

# ===================== 输出信息 =====================
print("RGA标签重构完成！")
print(f"网格划分: {GRID_SIZE}×{GRID_SIZE}")
print(f"瓶颈类别 ID: {sorted(BOTTLENECK_CLASSES)}")
```
#### 2.1. No Grid与2x2分别对应1.1和1.2的内容
#### 2.2. 4x4和6x6

模型为yaml/baseline+M1+M2.yaml，训练结果分别得到runs/detect/baseline+M1+M2(16)和runs/detect/baseline+M1+M2(16)

<img width="1178" height="335" alt="image" src="https://github.com/user-attachments/assets/5bf64e61-b529-43ab-8bf9-0f2bf663910a" />

<img width="1201" height="363" alt="image" src="https://github.com/user-attachments/assets/8a1e7def-0654-493f-93b9-e53797fd8f39" />

#### 2.3. 总结

2×2 网格划分的 RGA 标签重构效果最优，相比无网格基线，mAP50 从 0.73 提升至 0.821，同时推理速度也有提升，而更高的 4×4、6×6 网格划分会因标签冗余或特征过碎导致精度下降。


<img width="346" height="151" alt="image" src="https://github.com/user-attachments/assets/6fa3b1b1-e9d2-4f11-b3f3-11187df44b68" />


### 3. NEU-DET 数据集上θ的敏感性分析

在1.1中各类别标签进行评估后的Recall为：['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']=[0.182,0.745,0.884,0.721,0.632,0.951],因此θ取值为[0.5,0.7,0.9]。

RGA重构选择的网格数都是为2x2。

#### 3.1. θ为0.5与1.3对应。

#### 3.2. θ为0.7

进行RGA重构的有crazing,rolled-in_scale。
<img width="1182" height="352" alt="image" src="https://github.com/user-attachments/assets/aa5c294e-d903-4f07-ac63-7239092b32d8" />

#### 3.3. θ为0.9

进行RGA重构的有crazing,inclusion,patches,pitted_surface,rolled-in_scale。
<img width="1180" height="352" alt="image" src="https://github.com/user-attachments/assets/c8a84458-daa0-4324-bea4-2a365dde5f8a" />

#### 3.4. 总结

在NEU-DET数据集上，随着RGA重构阈值θ的提升（从0.5到0.9），缺陷标签数量显著增加，模型mAP50也随之从0.821稳步提升至0.921，同时FPS也保持在较高水平，说明基于缺陷召回率的RGA标签重构策略在该数据集上能有效强化低召回缺陷的监督信号，显著提升检测性能。


<img width="355" height="121" alt="image" src="https://github.com/user-attachments/assets/4c354aac-0d6c-4d5c-864e-b51e88a2472f" />

# 二、对数据集GC10-DET的实验

## **实验环境与参数**

与实验一不同的是，硬件采用 NVIDIA RTX 4080 SUPER (32GB)。由于数据集变大，为了减少训练时间，训练参数如下：训练共 80 轮，输入图像尺寸 1280，采用 SGD 优化器，关闭马赛克、随机擦除等复杂数据增强，开启混合精度与内存缓存，配合梯度累积、线程调优。

## **实验结果**

### 1. 对数据集的处理

- 标签文件名称并没有采用缺陷的名称命名，因此在对数据集进行划分时采用标签文件第一个缺陷来标记该标签文件的类型
- 由于训练时标签序号为7，8，9的数量较低，在训练时表现欠佳，因此进行删除

### 2. GC10-DET 数据集上不同模块的消融研究（M1：RGA；M2：MDLA；M3：BiFN）		

#### 2.1. baseline

<img width="1181" height="363" alt="image" src="https://github.com/user-attachments/assets/e00b8545-cfaa-4d82-85ed-a9171a3b4bc1" />

#### 2.2. baseline+M1

对2.1中模型训练得到的结果Recall<0.5进行RGA重构，由于部分缺陷在图片中的占比太小，而进行RGA重构的维度为2×2，这就表明有可能多个缺陷被划分在一个区域中，所以重构后缺陷数量可能会减少。为了避免重构后的重复标签，这里采用的去重处理。代码如下：
```python
import os
# ===================== 配置 =====================
BOTTLENECK_CLASSES = {1,5,6}   # 瓶颈类别
GRID_SIZE = 8                      # n×n划分
OVERLAP_TH = 0.1                     # 交集比例阈值
EPS = 1e-6

# 原始标签路径
INPUT_TRAIN = "./yolo_dataset/labels/train_original"
INPUT_VAL   = "./yolo_dataset/labels/val_original"
INPUT_TEST  = "./yolo_dataset/labels/test_original"

# 输出路径
OUTPUT_TRAIN = "./yolo_dataset/labels/train"
OUTPUT_VAL   = "./yolo_dataset/labels/val"
OUTPUT_TEST  = "./yolo_dataset/labels/test"

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_VAL, exist_ok=True)
os.makedirs(OUTPUT_TEST, exist_ok=True)
# ===================== 标签重构 =====================
def rga_reform_grid(txt_path, save_path):

    with open(txt_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
    bboxes=[]
    for line in lines:
        line=line.strip()
        if not line:
            continue
        cls,x,y,w,h=map(float,line.split())
        bboxes.append(
            (int(cls),x,y,w,h)
        )
    new_lines=[]
    grid_w=1.0/GRID_SIZE
    grid_h=1.0/GRID_SIZE
    # 用来记录哪些目标被成功映射
    hit_index=set()
    # ===================== 遍历所有网格 =====================
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            gx=grid_w*(j+0.5)
            gy=grid_h*(i+0.5)
            gw=grid_w
            gh=grid_h
            gx1=gx-gw/2
            gy1=gy-gh/2
            gx2=gx+gw/2
            gy2=gy+gh/2
            matched_cls=[]
            # ===================== 遍历所有缺陷 =====================
            for idx,(cls,x,y,w,h) in enumerate(bboxes):
                if cls not in BOTTLENECK_CLASSES:
                    continue
                bx1=x-w/2
                by1=y-h/2
                bx2=x+w/2
                by2=y+h/2
                # 计算交集
                ix1=max(gx1,bx1)
                iy1=max(gy1,by1)
                ix2=min(gx2,bx2)
                iy2=min(gy2,by2)
                iw=max(0,ix2-ix1+EPS)
                ih=max(0,iy2-iy1+EPS)
                inter=iw*ih
                box_area=w*h
                # 防止除0
                if box_area<EPS:
                    continue
                ratio=inter/box_area
                # 满足重叠阈值
                if ratio>=OVERLAP_TH:
                    matched_cls.append(cls)
                    hit_index.add(idx)
            # 去重
            matched_cls=list(set(matched_cls))
            # 保存多个类别
            for c in matched_cls:
                new_lines.append(
                    f"{c} "
                    f"{gx:.6f} "
                    f"{gy:.6f} "
                    f"{gw:.6f} "
                    f"{gh:.6f}"
                )
    # =====================
    # 未命中的瓶颈目标保留原标签
    # 防止标签彻底消失
    # =====================
    for idx,(cls,x,y,w,h) in enumerate(bboxes):
        if cls in BOTTLENECK_CLASSES:
            if idx not in hit_index:
                new_lines.append(
                    f"{cls} "
                    f"{x:.6f} "
                    f"{y:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}"
                )
        else:
            # 非瓶颈类原样保留
            new_lines.append(
                f"{cls} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}"
            )

    with open(save_path,'w',encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
# ===================== 批量处理 =====================
def process_folder(input_dir,output_dir):
    files=os.listdir(input_dir)
    for fname in files:
        if fname.endswith(".txt"):
            rga_reform_grid(
                os.path.join(input_dir,fname),
                os.path.join(output_dir,fname)
            )
process_folder(INPUT_TRAIN,OUTPUT_TRAIN)
process_folder(INPUT_VAL,OUTPUT_VAL)
process_folder(INPUT_TEST,OUTPUT_TEST)
print("标签重构完成")
print("GRID:",GRID_SIZE)
print("OVERLAP_TH:",OVERLAP_TH)
print("瓶颈类别:",BOTTLENECK_CLASSES)
```
<img width="1188" height="357" alt="image" src="https://github.com/user-attachments/assets/6bd42274-8824-4a39-88ec-6a7e6b335f4e" />

#### 2.3. baseline+M1+M2
<img width="1187" height="355" alt="image" src="https://github.com/user-attachments/assets/90668044-e291-4506-b352-2ccab905de94" />

#### 2.4. baseline+M2

在后续的实验中，发现进行重构后的训练结果欠佳，因此将不采用重构。

<img width="1180" height="360" alt="image" src="https://github.com/user-attachments/assets/82ba92fd-c01b-47ae-98d9-8950ff29e03f" />

#### 2.5. baseline++M2+M3(conv)

<img width="1168" height="361" alt="image" src="https://github.com/user-attachments/assets/e1b3d363-1b51-4ed0-9331-eb740556acb3" />

#### 2.6. baseline+M2+M3(scdown)

<img width="1168" height="371" alt="image" src="https://github.com/user-attachments/assets/10cc7967-e954-45f1-b7de-cc5b17c20acf" />

#### 2.7. 总结

在GC10-DET数据集上，单独引入RGA（M1）会造成精度下降，而结合MDLA（M2）后可实现精度回升，且MDLA+Conv的组合在该数据集上表现最优；BiFN（M3）的两种实现方式均未带来稳定增益，SCDown甚至会导致mAP50大幅降低，说明所提模块在不同缺陷场景下的泛化性存在差异。

<img width="850" height="217" alt="image" src="https://github.com/user-attachments/assets/ba1e8340-9e03-4ed4-80a5-9b6c9a3738ed" />

### 3. GC10-DET 数据集上的网格实验					

#### 3.1. No Grid和2×2分别对应2.1和2.2

#### 3.2. 4×4、6×6和8×8
<img width="1181" height="355" alt="image" src="https://github.com/user-attachments/assets/3fa1a778-439a-413b-9c95-3a3b2889b978" />

<img width="1172" height="367" alt="image" src="https://github.com/user-attachments/assets/a1d7282a-d80b-4aa9-bc8c-c2a52f81fadf" />

<img width="1173" height="360" alt="image" src="https://github.com/user-attachments/assets/f1446e84-58c9-4ee9-99ed-befa28897118" />

#### 3.3. 总结

在 GC10-DET 数据集上，随着网格尺寸增大，缺陷标签数量虽显著增加，但模型 mAP50 却持续下降，相比无网格基线，8×8 网格下精度降幅超 50%，说明 RGA 标签重构在该数据集上反而带来了性能损失，不具备跨数据集的普适性。
<img width="589" height="169" alt="image" src="https://github.com/user-attachments/assets/d0a46b30-6580-4573-a72a-d6c74c939350" />

### 4. GC10-DET 数据集上θ的敏感性分析			

在2.1中各类别标签进行评估后的Recall为：['1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban', '5_youban', '6_siban', '7_yiwu']=[1,0.26,0.956,0.588,0.638,0.286,0.247],因此θ取值为[0.5,0.7]。

RGA重构选择的网格数都是为2x2。

#### 4.1. θ为0.5与2.2对应

#### 4.2. θ为0.7

<img width="1177" height="371" alt="image" src="https://github.com/user-attachments/assets/963960c1-bdfc-4340-b657-662d3880f658" />

#### 4.3. 总结

在GC10-DET数据集上，RGA标签重构阈值θ的增大（从0.5提升至0.7）会导致模型mAP50显著下降，精度损失明显，说明以缺陷召回率为阈值的RGA策略在该数据集上不仅无法提升性能，反而会引入噪声标签，对模型产生负面影响。

<img width="355" height="97" alt="image" src="https://github.com/user-attachments/assets/edf38b74-1bc3-4587-ad08-59102697792f" />

