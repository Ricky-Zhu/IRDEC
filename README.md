# IRDEC

This repository contains the official implementation for IRDEC introduced in the following paper:

**Learning to Solve Tasks with Exploring Prior Behaviours**

[Ruiqi Zhu](https://ricky-zhu.github.io/), [Siyuan Li](https://siyuanlee.github.io/), [Tianhong Dai](https://tianhongdai.xyz/),
[Chongjie Zhang](http://people.iiis.tsinghua.edu.cn/~zhang/), [Oya Celiktutan](https://nms.kcl.ac.uk/oya.celiktutan/)

## Citation
If you find our work useful in your research, please cite:

## Environment
- Python 3.7
- Other required python packages are specified by `requirements.txt`.

## Prerequisites
- Install Mujoco and other required packages. 
- Register Wandb account.
- Clone this repository.

## Reproducing Ant-Umaze Experiment
```shell
chmod +x ant_umaze_train.sh
./ant_umaze_train.sh
```

## Reproducing Ant-FourRoom Experiment
```shell
chmod +x ant_four_room_train.sh
./ant_four_room_train.sh
```