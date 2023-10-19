## 摘要
:arrow_right:：本文档为 YOLOv5 的基础教程文档
:warning:：本文档将不涉及任何关于算法的内容

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

2. 打开 Terminal（终端） 或者 CMD（命令行）并输入以下指令来查看版本：
    ```bash
    conda -V # 查看 Anaconda 版本
    ```
    如果出现类似于 `conda 23.7.4` 的输出则说明安装成功了。 :triangular_flag_on_post:

- ### 搭建 YOLO 环境

1. <span id="jump"></span>在 Terminal（终端）或者 CMD（命令行）中输入以下指令来查看环境：
    ```bash
    conda env list # 查看 Anaconda 环境
    ```
    此时应该只存在一个 `base` 环境：
    ```bash
    # conda environments:
    #
    base                    /Users/Stewart222b/anaconda3
    ```

2. 输入以下指令来为 YOLO 创建一个新的环境：
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
    执行完毕后，Terminal（终端） 或者 CMD（命令行）的命令输入行应该变成了类似于以下的样子：
    ```
    (yolo) Stewart222b@This-MacBook-Pro ~ % 
    ```
    如果想要退出环境，可以直接关掉 Terminal（终端） 或者 CMD（命令行）的窗口（不推荐）或者输入以下指令：
    ```bash
    conda deactivate yolo # 退出 yolo 环境
    ```

    现在，我们已经完成了安装 YOLOv5 之前的准备工作了。:triangular_flag_on_post:

- ### 配置 YOLOv5
    :triangular_flag_on_post:

- ### 配置 LabelImg
    :triangular_flag_on_post:

---

## 数据准备

- ### 数据获取
    :triangular_flag_on_post:
- ### 数据增强
    :triangular_flag_on_post:
- ### 数据标注
    :triangular_flag_on_post:
- ### 数据校验
    :triangular_flag_on_post:
- ### 数据分配
    :triangular_flag_on_post:

---

## 模型训练

- ### 本地训练
    :triangular_flag_on_post:
- ### 线上训练
    :triangular_flag_on_post:
- ### 结果分析
    :triangular_flag_on_post:

---

## 模型量化

- ### 量化工具
    :triangular_flag_on_post:

---

## 模型部署

- ### 模型转换
    :triangular_flag_on_post:
    :triangular_flag_on_post:

---