<p align="center">
    <!-- pypi-strip -->
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/yanglijie-dev/mmPPT/blob/main/code/mmppt/docs/overall_structure.pdf">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/XXXX/XXXX/main/docs/logo.png">
    <!-- /pypi-strip -->
    <img alt="mmPPT" src="https://raw.githubusercontent.com/XXXXX/XXXXX/main/docs/logo.png" width="400">
    <!-- pypi-strip -->
    </picture><br>
    <!-- /pypi-strip -->
</p>


This repository is an official implementation of the following paper:

- **mmPPT: Hierarchical-serialization-enhanced point transformer for mmWave pedestrian reconstruction,**  *Lijie Yang, Yu Wang, Zhaohui Yang, Tongkai Xu, Lizhi Dang, Zhongyue Chen, Weipeng Mao,*  Information Fusion, Volume 127, Part B, 2026, 103835, ISSN 1566-2535.
[[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253525008978) ]  [ [Project](https://github.com/yanglijie-dev/mmPPT) ] 

- Datasets:
[mmBody](https://github.com/Chen3110/mmBody)


## Citation
If you find _mmPPT_ useful to your research, please cite our work.​
```
@article{YANG2026103835,
title = {mmPPT: Hierarchical-serialization-enhanced point transformer for mmWave pedestrian reconstruction},
journal = {Information Fusion},
volume = {127},
pages = {103835},
year = {2026},
issn = {1566-2535},
doi = {https://doi.org/10.1016/j.inffus.2025.103835},
url = {https://www.sciencedirect.com/science/article/pii/S1566253525008978},
author = {Lijie Yang and Yu Wang and Zhaohui Yang and Tongkai Xu and Lizhi Dang and Zhongyue Chen and Weipeng Mao}
}
```

## Overview

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)

## Installation

### Requirements
- Ubuntu: 18.04 and above.
- CUDA: 11.3 and above.
- PyTorch: 1.10.0 and above.

### Conda Environment

```bash
conda create -n mmppt python=3.8 -y
conda activate mmppt
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# PPT (clip)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d
```

## Data Preparation

### mmBody

- Download the [mmBody](https://github.com/Chen3110/mmBody) dataset extract data to the "data" folder in the mmPPT project, e.g. ```./data/mmBody```

The data structure should be like this: 
```
.
├──data
    ├──mmBody
        ├── test/
        │   ├── furnished/
        │   │   ├── sequence_0/
        │   │   │   ├── calib.txt
        │   │   │   ├── depth/
        │   │   │   ├── image/
        │   │   │   ├── mesh/
        │   │   │   └── radar/
        │   │   └── sequence_1/
        │   ├── lab1/
        │   │   ├── sequence_0/
        │   │   └── sequence_1/
        │   ├── lab2/
        │   │   ├── sequence_0/
        │   │   └── sequence_1/
        │   ├── occlusion/
        │   │   ├── sequence_0/
        │   │   └── sequence_1/
        │   ├── poor_lighting/
        │   │   ├── sequence_0/
        │   │   └── sequence_1/
        │   ├── rain/
        │   │   ├── sequence_0/
        │   │   └── sequence_1/
        │   └── smoke/
        │       ├── sequence_0/
        │       └── sequence_1/
        └── train/
            ├── sequence_0/
            │   ├── calib.txt
            │   ├── depth/
            │   ├── image/
            │   ├── mesh/
            │   └── radar/
            ├── sequence_1/
            ├── sequence_2/
            ├── sequence_3/
            ├── sequence_4/
            ├── sequence_5/
            ├── sequence_6/
            ├── sequence_7/
            ├── sequence_8/
            ├── sequence_9/
            ├── sequence_10/
            ├── sequence_11/
            ├── sequence_12/
            ├── sequence_13/
            ├── sequence_14/
            ├── sequence_15/
            ├── sequence_16/
            ├── sequence_17/
            ├── sequence_18/
            └── sequence_19/
```
	
## SMPL file Preparation

## Quick Start

### Training
**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard, and checkpoints will also be saved into the experiment folder during the training process.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Direct (Recommended)
export PYTHONPATH=./
python ./tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
# Script
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}

```

For example:
```bash
# Direct (Recommended)
conda activate mmppt && \
export PYTHONPATH=${proj_root}/code/mmppt:$PYTHONPATH && \
cd ${proj_root}/code/mmppt && \
python ./tools/train.py --config-file ./configs/mmBody/mmppt-base.py --options save_path="../output"

# By script
# -p is default set as python and can be ignored
conda activate mmppt && cd ${proj_root}/code/mmppt && sh scripts/train.sh -p python -g 1 -d mmBody -c mmppt-base -n my_exp
```
### Testing

```bash
# Direct
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}
```
For example:
```bash
# Direct
conda activate mmppt && \
export PYTHONPATH=${proj_root}/code/mmppt:$PYTHONPATH && \
cd ${proj_root}/code/mmppt && \
python ./tools/test.py --config-file ./configs/mmBody/mmppt-base.py --options save_path="../output" weight=../output/model_ckpt.pth
```
