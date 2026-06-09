# BEVFusion-Temporal: Multi-Modal 3D Object Detection with Temporal Multi-Frame Fusion

<div align="center">

Undergraduate Graduation Thesis, School of Information Science and Engineering, Northeastern University (China)



[BEVFusion-Temporal](https://github.com/ZhanRuiXin/BEVFusion-Temporal)  |  [Thesis PDF](doc/202606081903.pdf)

（ [简体中文](README.md) | English ）

</div>

## Abstract

Multi-modal fusion is a key technology for accurate and reliable perception in autonomous driving systems. BEVFusion unifies multi-modal features in BEV space, effectively preserving both geometric and semantic information. However, it relies solely on single-frame information, struggling with occluded targets, distant objects, and other complex scenarios.

To address these issues, we propose BEVFusion-Temporal, a temporal multi-frame fusion method based on [BEVFusion](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion). By storing and aligning multi-frame BEV features, our method effectively integrates historical information to enhance detection capability for occluded and distant targets. Experimental results show improvements in key metrics such as mAP and NDS compared to the baseline BEVFusion.

## Introduction

This project is built upon the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) framework and the [MIT Han Lab BEVFusion](https://github.com/mit-han-lab/bevfusion) project. Key improvements include:

- Historical BEV feature caching
- Ego-motion based coordinate alignment
- Channel concatenation with 1×1 convolution fusion

## Environment Setup

Follow the [MMDetection3D official guide](https://mmdetection3d.readthedocs.io/en/latest/get_started.html) to install the environment.

Alternatively, use the provided Dockerfile to build an image (Docker version >= 19.03):

```bash
docker build -t mmdet3d docker/
```

Run the Docker container:

```bash
docker run -it -v /to/your/path/BEVFusion-Temporal:/to/your/workplace/BEVFusion-Temporal --name BEVFusion --shm-size 32g --gpus all mmdet3d /bin/bash
```

Access the container shell:

```bash
docker start BEVFusion
docker exec -it BEVFusion /bin/bash
```

Install MMDetection3D 1.4.0:

```bash
cd /to/your/path/BEVFusion-Temporal
pip install -v -e . 
```

Compile CUDA operators:

```bash
python projects/BEVFusion/setup.py develop
```

## Data Preparation

Download the complete dataset from the [nuScenes](https://www.nuscenes.org/).


This project also provides a Python script for downloading and extracting the complete dataset from the official website. If needed, please register an account on the official website and replace the account credentials in the script before downloading.

Run the following command from the project root to download the dataset to the designated path. If you wish to store the dataset elsewhere, modify the path in the script and then create a symlink to the project's expected path.

```bash
python nuscenes_downloader/downloader.py
```

After data preparation, you will be able to see the following directory structure:
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

Generate dataset index files:

```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-trainval
```

Download pre-trained weights:

```bash
wget https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth -P pretrained/
```

## Training

Stage 1: Train LiDAR-only detector

```bash
bash tools/dist_train.sh projects/BEVFusion/configs/BEVFusionTemporal_lidar.py 1
```

Stage 2: Train multi-modal fusion model

```bash
bash tools/dist_train.sh projects/BEVFusion/configs/BEVFusionTemporal_lidar-cam.py 1 --cfg-options load_from=${LIDAR_PRETRAINED_CHECKPOINT} model.img_backbone.init_cfg.checkpoint="pretrained/swint-nuimages-pretrained.pth"
```

To enable mixed precision training to reduce memory usage, add the `--amp` flag.

If you need to change the number of GPUs or specify particular GPUs, modify the command arguments accordingly. Also make sure to adjust the corresponding training parameters in the config files under `projects/BEVFusion/configs` after carefully reviewing them.

## Testing

Test LiDAR-only model

```bash
bash tools/dist_test.sh projects/BEVFusion/configs/BEVFusionTemporal_lidar.py ${LIDAR_CHECKPOINT} 1
```

Test multi-modal fusion model

```bash
bash tools/dist_test.sh projects/BEVFusion/configs/BEVFusionTemporal_lidar-cam.py ${FUSION_CHECKPOINT} 1
```

Replace `${LIDAR_CHECKPOINT}` and `${FUSION_CHECKPOINT}` with the actual `.pth` checkpoint paths (located in the `work_dir` folder by default). To specify GPUs, add `--gpu-ids 0` or similar arguments.

## Experimental Results

Experiments were conducted on a single NVIDIA RTX 4090 GPU. Results on the test set are shown below (for reference only):

| Method | NDS (%) | mAP (%) |
| :--- | :---: | :---: |
| BEVFusion (baseline) | 70.25 | 66.27 |
| BEVFusion-Temporal (Ours) | 71.17 | 67.02 |