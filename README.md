# Paddle-SSPCAB

## 目录

- [1. 简介]()
- [2. 数据集]()
- [3. 复现精度]()
- [4. 模型数据与环境]()
    - [4.1 目录介绍]()
    - [4.2 准备环境]()
    - [4.3 准备数据]()
- [5. 开始使用]()
    - [5.1 模型训练]()
    - [5.2 模型评估]()
    - [5.3 模型预测]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 模型信息]()

## 1. 简介
在成功的异常检测方法中，有一类方法依赖于对被mask掉信息的预测并利用与被mask信息相关的重建误差作为异常分数。与相关方法不同，文章提出将基于重建的功能集成到一个新的自监督的预测体系结构模块中。作者主要是通过一个带有扩张卷积的卷积层进行卷积，然后将结果通过通道注意力模块。提出的自监督块是通用的，可以很容易地纳入各种最新的异常检测方法。

[aistudio在线运行](https://aistudio.baidu.com/aistudio/projectdetail/4398039)

**论文:** [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/pdf/2104.04015v1.pdf)

**论文:** [Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection](https://arxiv.org/pdf/2111.09099.pdf)

**参考repo:** [sspcab](https://github.com/ristea/sspcab)

**参考repo:** [pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste)


## 2. 数据集

MVTec AD是MVtec公司提出的一个用于异常检测的数据集。与之前的异常检测数据集不同，该数据集模仿了工业实际生产场景，并且主要用于unsupervised anomaly detection。数据集为异常区域都提供了像素级标注，是一个全面的、包含多种物体、多种异常的数据集。数据集包含不同领域中的五种纹理以及十种物体，且训练集中只包含正常样本，测试集中包含正常样本与缺陷样本，因此需要使用无监督方法学习正常样本的特征表示，并用其检测缺陷样本。

数据集下载链接：[AiStudio数据集](https://aistudio.baidu.com/aistudio/datasetdetail/116034) 解压到data文件夹下


## 3. 复现精度

| defect_type   |   CutPaste(3-way)+SSPCAB(复现) |   CutPaste(3-way)+SSPCAB |  CutPaste (3-way) |
|:--------------|--------------------:|-------------------:|-----------------------------:|
| bottle        |                100.0 |               98.6 |                         98.3 |
| cable         |                90.7 |               82.9 |                         80.6 |
| capsule       |                93.0 |               98.1 |                         96.2 |
| carpet        |                90.1 |               90.7 |                         93.1 |
| grid          |                100.0 |               99.9 |                         99.9 |
| hazelnut      |                99.6 |               98.3 |                         97.3 |
| leather       |               100.0 |              100.0 |                        100.0 |
| metal_nut     |                98.2 |              100.0 |                         99.3 |
| pill          |                94.8 |               95.3 |                         92.4 |
| screw         |                81.1 |               90.8 |                         86.3 |
| tile          |                99.3 |               94.0 |                         93.4 |
| toothbrush    |               99.4 |               98.8 |                         98.3 |
| transistor    |                98.5 |               96.5 |                         95.5 |
| wood          |                100.0 |               99.2 |                         98.6 |
| zipper        |               100.0 |               98.1 |                         99.4 |
| average       |                96.3 |               96.1 |                         95.2 |


## 4. 模型数据与环境

### 4.1 目录介绍

```
    |--images                         # 测试使用的样例图片，两张
    |--deploy                         # 预测部署相关
        |--export_model.py            # 导出模型
        |--infer.py                   # 部署预测
    |--data                           # 训练和测试数据集
    |--lite_data                      # 自建立的小数据集，含有bottle
    |--logdirs                        # 训练train和测试eval打印的日志信息  
    |--eval                           # eval输出文件
    |--models                         # 训练的模型权值
    |--test_tipc                      # tipc代码
    |--tools                          # 工具类文件
        |--cutpaste.py                # 论文代码
        |--dataset.py                 # 数据加载
        |--density.py                 # 高斯聚类代码
        |--model.py                   # 论文模型
    |--predict.py                     # 预测代码
    |--eval.py                        # 评估代码
    |--train.py                       # 训练代码
    |----README.md                    # 用户手册
```

### 4.2 准备环境

- 框架：
  - PaddlePaddle >= 2.3.1
- 环境配置：使用`pip install -r requirement.txt`安装依赖。


### 4.3 准备数据

- 全量数据训练：
  - 数据集下载链接：[AiStudio数据集](https://aistudio.baidu.com/aistudio/datasetdetail/116034) 解压到data文件夹下
- 少量数据训练：
  - 无需下载数据集，直接使用lite_data里的数据
  
## 5. 开始使用
### 5.1 模型训练

- 全量数据训练：
  - `python train.py --type all --batch_size 96 --test_epochs 10 --head_layer 1 --seed 102`
- 少量数据训练：
  - `python train.py --data_dir lite_data --type bottle --epochs 10 --test_epochs 5 --batch_size 5`
  
模型训练权重全部保存在models文件下

- 部分训练日志如下所示：
```
Type : bottle Train [ Epoch 1/500 ], loss: 1.3306, avg_reader_cost: 1.2196 avg_batch_cost: 3.4922, avg_ips: 27.4898.
Type : bottle Train [ Epoch 2/500 ], loss: 0.9671, avg_reader_cost: 1.0007 avg_batch_cost: 1.3481, avg_ips: 71.2094.
Type : bottle Train [ Epoch 3/500 ], loss: 0.7691, avg_reader_cost: 1.1231 avg_batch_cost: 1.4709, avg_ips: 65.2671.
Type : bottle Train [ Epoch 4/500 ], loss: 0.6155, avg_reader_cost: 1.2133 avg_batch_cost: 1.5619, avg_ips: 61.4622.
Type : bottle Train [ Epoch 5/500 ], loss: 0.5655, avg_reader_cost: 1.1475 avg_batch_cost: 1.4872, avg_ips: 64.5512.
``` 
模型训练日志全部保存在logdirs文件下

可以将训练好的模型权重[下载](https://aistudio.baidu.com/aistudio/datasetdetail/162384) 解压为models文件放在本repo/下，直接对模型评估和预测

### 5.2 模型评估(通过5.1完成训练后)

- 全量数据模型评估：`python eval.py --type all --data_dir data --head_layer 8 --density paddle`
- 少量数据模型评估：`python eval.py --data_dir lite_data --type bottle`

评估会生成验证结果保存在项目evel文件下

### 5.3 模型预测（需要预先完成5.1训练以及5.2的评估）

- 模型预测：`python predict.py --data_type bottle --img_file images/good.png`

结果如下：
```
预测结果为：正常 预测分数为：26.2237
```

- 基于推理引擎的模型预测：
```
python deploy/export_model.py
python deploy/infer.py --data_type bottle --img_path images/good.png
```
结果如下：
```
> python deploy/export_model.py
inference model has been saved into deploy

> python deploy/infer.py --data_type bottle --img_path images/good.png
image_name: images/good.png, data is normal, score is 26.223722457885742, threshold is 51.2691650390625
```


## 6. 自动化测试脚本
- tipc 所有代码一键测试命令（少量数集）
```
bash test_tipc/test_train_inference_python.sh test_tipc/configs/resnet18/train_infer_python.txt lite_train_lite_infer 
```

结果日志如下
```
[Run successfully with command - python3.7 train.py --type bottle --test_epochs 3 --model_dir=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0 --epochs=2   --batch_size=1!]
[Run successfully with command - python3.7 eval.py --type bottle --pretrained=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0/model-bottle.pdparams! ]
[Run successfully with command - python3.7 deploy/export_model.py  --pretrained=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0/model-bottle.pdparams --save-inference-dir=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0!  ]
[Run successfully with command - python3.7 deploy/infer.py --use-gpu=True --model-dir=./log/resnet18/lite_train_lite_infer/norm_train_gpus_0 --batch-size=1   --benchmark=False > ./log/resnet18/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 !  ]
```

## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 模型信息

| 信息 | 描述 |
| --- | --- |
| 作者 | Lieber|
| 日期 | 2022年8月 |
| 框架版本 | PaddlePaddle==2.3.1 |
| 应用场景 | 异常检测 |
| 硬件支持 | GPU、CPU |
| 在线体验 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/4398039)
