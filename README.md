# BEVFusion-Temporal: 基于时序多帧融合的多模态3D目标检测
</div>

<div align="center">

东北大学信息科学与工程学院本科生毕业设计（论文）

[BEVFusion-Temporal](https://github.com/ZhanRuiXin/BEVFusion-Temporal)  |  [论文 PDF](doc/202606081903.pdf)

（ 简体中文 | [English](README_en-US.md) ）

</div>



## 摘要

多模态融合是自动驾驶系统实现精准可靠感知的关键技术。BEVFusion将多模态特征统一到BEV空间，有效保留了几何与语义信息，但其仅依赖单帧信息，难以处理遮挡目标、远距离物体等复杂场景。

针对上述问题，本文提出BEVFusion-Temporal，一种基于 [BEVFusion](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion) 的时序多帧融合方法。通过存储并对齐多帧BEV特征，有效整合历史信息，增强对遮挡和远距离目标的检测能力。实验结果表明，与基线BEVFusion相比，本文方法在mAP和NDS等关键指标上均有提升。

## 简介

本项目基于 [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) 框架，以 [MIT Han Lab 的 BEVFusion 项目](https://github.com/mit-han-lab/bevfusion) 为基础，实现了BEVFusion-Temporal时序多帧融合方法，主要改进包括：

- 历史BEV特征缓存
- 基于自车运动的坐标对齐
- 通道拼接与1×1卷积融合

## 环境准备
按照 [MMDetecrion3D官方手册](https://mmdetection3d.readthedocs.io/zh-cn/latest/get_started.html) 安装环境，并进行相应的环境验证工作。

或是使用本项目提供的Dockerfile来构建一个镜像，这也是本项目相关论文的实验环境。请确保您的docker版本 >= 19.03。
```bash
docker build -t mmdet3d docker/
```
用以下命令运行 Docker 镜像：
```bash
docker run -it -v /to/your/path/BEVFusion-Temporal:/to/your/workplace/BEVFusion-Temporal --name BEVFusion --shm-size 32g --gpus all mmdet3d /bin/bash
```
后续用shell访问容器的相关命令：
```bash
docker start BEVFusion
docker exec -it BEVFusion /bin/bash
```
安装MMDetection3D 1.4.0：
```bash
cd /to/your/path/BEVFusion-Temporal
pip install -v -e . 
```

编译CUDA算子：
```bash
python projects/BEVFusion/setup.py develop
```

## 数据准备
在 [nuScenes](https://www.nuscenes.org/) 官方网站下载完整的数据集。

下载后相关文件路径如下所示：
```
BEVFusion-Temporal
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
```

本项目中也提供了通过官方网站下载并解压完整数据集的python脚本，如有需要请在官网注册账号后通过该脚本进行下载。下载前请更换python脚本中账号密码的相关字段。

在当前项目路径下使用以下命令可以将数据集下载到本项目的指定路径，如果您需要将数据集存放在其他路径，请在脚本中更改路径，之后将数据集文件软链接到项目指定的路径上。
```bash
python nuscenes_downloader/downloader.py
```

生成数据集索引文件
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-trainval
```

下载预训练权重
```bash
wget https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth -P pretrained/
```

## 训练阶段
第一阶段：训练纯激光雷达检测器
```bash
bash tools/dist_train.sh projects/BEVFusion/configs/BEVFusionTemporal_lidar.py 1
```

第二阶段：训练多模态融合模型
```bash
bash tools/dist_train.sh projects/BEVFusion/configs/BEVFusionTemporal_lidar-cam.py 1 --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint="pretrained/swint-nuimages-pretrained.pth"
```
如需启用混合精度训练以减少显存占用，请使用`--amp`参数。

如果您需要修改使用GPU的数量，或是指定GPU，请自行修改命令中的参数，并在仔细阅读`projects/BEVFusion/configs`中的相关配置文件后修改相应的训练参数。

## 测试模型
测试纯激光雷达模型
```bash
bash tools/dist_test.sh projects/BEVFusion/configs/BEVFusionTemporal_lidar.py ${LIDAR_CHECKPOINT} 1
```

测试多模态融合模型
```bash
bash tools/dist_test.sh projects/BEVFusion/configs/BEVFusionTemporal_lidar-cam.py ${FUSION_CHECKPOINT} 1
```
将 `${LIDAR_CHECKPOINT}` 和 `${FUSION_CHECKPOINT}` 替换为实际的`.pth`权重文件路径（默认在`work_dir`文件夹中）。如需指定GPU，请在命令末尾添加 `--gpu-ids 0` 等参数。

## 评估数据
本项目在单卡 NVIDIA RTX 4090 下进行实验，在测试集中的评估结果如下，仅供参考：
| 方法 | NDS (%) | mAP (%) |
| :--- | :---: | :---: |
| BEVFusion (baseline) | 70.25 | 66.27 |
| BEVFusion-Temporal (Ours) | 71.17 | 67.02 |