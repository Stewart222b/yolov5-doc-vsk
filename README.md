## 摘要
:arrow_right:：本文档为 YOLOv5 的基础教程文档
:warning:：本文档将不涉及任何关于算法的内容

---

## 目录
- [YOLOv5 介绍](#yolov5-介绍)
- [安装](#安装)
    - 安装 Anaconda3
    - 搭建 YOLO 环境
    - 配置 YOLOv5
    - 配置 LabelImg
- [数据准备](#数据准备)
    - 数据获取
    - 数据增强
    - 数据标注
    - 数据校验
    - 数据分配
- [模型训练](#模型训练)
    - 本地训练
    - 线上训练
    - 结果分析
- [模型量化](#模型量化)
    - 量化工具
- [模型部署](#模型部署)
    - 模型转换

---

## YOLOv5 介绍

:triangular_flag_on_post:

---

## 安装

- ### Anaconda 安装
    <img src="./conda.jpeg" alt=conda width=50%>

1. 从 Anaconda 官网 [下载](https://www.anaconda.com/download)，如果网速慢可以从清华镜像源 [下载](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/) 并安装 Anaconda3 。

2. 打开 terminal（终端） 或者 cmd（命令行）并输入以下指令来查看版本：
    ```bash
    conda -V # 查看 Anaconda 版本
    ```
    如果出现类似于 `conda 23.7.4` 的输出则说明安装成功了。 :triangular_flag_on_post:

- ### 搭建 YOLO 环境

1. <span id="jump"></span>在 terminal 或者 cmd 中输入以下指令来查看环境：
    ```bash
    conda env list # 查看 Anaconda 环境
    ```
    此时应该只存在一个 `base` 环境：
    ```bash
    # conda environments:
    #
    base                    /Users/Stewart222b/anaconda3
    ```

2. 输入以下指令来为 yolo 创建一个新的环境：
    ```bash
    conda create -n yolo # 创建 Anaconda 环境
    ```
    `-n` 参数用来指定新环境的名称，上面的指令创建了一个名为 `yolo` 的环境。其中 python 版本默认为 `base` 环境中的 python 版本。如果对 python 版本有要求，可以在创建环境的时侯加上 `python=3.x` （ x 为任意版本）在指令末尾。
    
    <br>再次输入 [1.](#jump) 中的指令来查看环境，此时应该新增了一个 `yolo` 环境：
    ```bash
    # conda environments:
    #
    base                    /Users/Stewart222b/anaconda3
    yolo                    /Users/Stewart222b/anaconda3/envs/yolo
    ```
3. 输入以下指令可以切换到刚刚创建的 `yolo` 环境：
    ```bash
    conda activate yolo # 切换到 yolo 环境
    ```
    执行完毕后，terminal 或者 cmd 的命令输入行应该变成了类似于以下的样子：
    ```bash
    (yolo) Stewart222b@This-MacBook-Pro ~ % # terminal
    ```
    ```bash
    (yolo) C:\Projects> # cmd
    ```
    如果想要退出环境，可以直接关掉 terminal 或者 cmd 的窗口（不推荐）或者输入以下指令：
    ```bash
    conda deactivate yolo # 退出 yolo 环境
    ```

    现在 yolo 的环境就算配置好了。:triangular_flag_on_post:

- ### 配置 YOLOv5
1. 从 GitHub [下载](https://github.com/ultralytics/yolov5) 并解压 yolov5 源码或者直接使用以下指令克隆：
    ```bash
    git clone https://github.com/ultralytics/yolov5.git
    ```
    打开 `yolov5` 文件夹之后可以看到里面有个 `requirements.txt` 文件， 里面记录了需要安装的包。

2. 打开 terminal 或者 cmd 并切换到 `yolo` 环境，输入:
    ```bash
    pip install -r requirements.txt
    ```
    来在 `yolo` 环境下安装所需要的包。

    <br>安装完之后就可以准备数据然后开始模型的训练了。本文档提供了一个 demo 数据集可供训练，如果想直接训练模型可以跳转到 [模型训练](#模型训练) 。:triangular_flag_on_post:

---

## 数据准备

- ### 数据获取
    :triangular_flag_on_post:
- ### 数据标注
    :warning: 一定要筛查整个数据集后再确定标注的类别
    :warning: 制定一个统一的标注标准：如果为多人同时标注一个数据集，尽量使用一个标准。（举例：有车被挡住了，是只标注露出的部分还是连同被遮挡着部分一同标注）
    :triangular_flag_on_post:
- ### 数据增强
    如果想提升模型在不同场景和环境下的性能，数据增强是十分重要的。同时，数据增强可以帮我们获得额外的数据。
    :triangular_flag_on_post:
- ### 数据校验
    检查标注完的数据集中样本数量的分配，确保每个类别的样本数量不能过少。 比如 有一个要区分
    :triangular_flag_on_post:
- ### 数据分配
    训练集和验证集的比例建议为 9:1 或者 8:2 。
    
    假设现在有 1000 张数据，在其中随机抽取 200 张作为验证集。但是不能完全随机，要保证验证集涉及到训练集中的所有场景和类别。比如数据集中一共有20个类，但是验证集中却只出现了十五个类，这是十分致命的，并且会严重影响训练结果。

    :triangular_flag_on_post:

---

## 模型训练

demo 数据集内容：50 张哆啦A梦的图片。类别仅有 1 类：A Meng

demo 数据集在这里 [下载](https://www.aliyundrive.com/s/hz6un5Kd9T5) ，提取码：`e28r`


- ### 本地训练
    路径跳转到 `yolov5` 文件夹
    ```bash
    cd C:/projects/yolo/yolov5 # 位置修改一下
    ```
    训练指令
    ```bash
    python train.py --batch -1 --epoch 100 --weights yolov5s.pt --data ./data/A.yaml
    ```
    :triangular_flag_on_post:
- ### 线上训练
    除了在自己的设备上训练，也可以在服务器上来训练模型。一些线上训练的优缺点：
    - 优点：
        - 速度快：服务器上的显卡大多性能强大、算力高，如 RTX4090 等。
        - 不占资源：线上训练不影响本地设备的性能。
    - 缺点：
        - 收费：需要花钱租用显卡，费用大概几块钱一小时。
        - 时间受限：在高峰期可能出现没有显卡的情况。

    <br>以一个平台 [AutoDL算力云](https://www.autodl.com/home) 为例，介绍一下使用流程：

    <br>简单使用流程：注册账号 :arrow_right: 充值 :arrow_right: 租用显卡 :arrow_right: 移植环境 :arrow_right: 数据上传 :arrow_right: 开始训练



    :triangular_flag_on_post:

- ### 结果分析
    检测指令
    ```bash
    python detect.py --weights ./runs/train/exp2/weights/best.pt --img 640 --conf 0.25 --source ../test2.jpg --save-txt
    ```
    :triangular_flag_on_post:
---

## 模型量化

- ### 量化工具
    导出为 onnx 格式
    ```bash
    python export.py
    ```
    :triangular_flag_on_post:

---

## 模型部署

- ### 模型转换
    :triangular_flag_on_post:
    :triangular_flag_on_post:

---