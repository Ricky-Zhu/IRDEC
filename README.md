# Learning to Solve Tasks with Exploring Prior Behaviours

This repository contains the official implementation for IRDEC introduced in the following paper:

[**Learning to Solve Tasks with Exploring Prior Behaviours**](https://arxiv.org/abs/2307.02889)

[Ruiqi Zhu](https://ricky-zhu.github.io/), [Siyuan Li](https://siyuanlee.github.io/), [Tianhong Dai](https://tianhongdai.xyz/),
[Chongjie Zhang](http://people.iiis.tsinghua.edu.cn/~zhang/), [Oya Celiktutan](https://nms.kcl.ac.uk/oya.celiktutan/)


## Environment
- Python 3.7
- Other required python packages are specified by `requirements.txt`.

## Prerequisites
- Install Mujoco and other required packages. 
- Register Wandb account.
- Clone this repository.

## Troubleshooting
if meet `RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED`, which is caused due to the cuda toolkit.

```shell
pip3 install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```


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

## Citation
If you find this code useful in your research, please consider citing:

```
@article{zhu2023learning,
  title={Learning to Solve Tasks with Exploring Prior Behaviours},
  author={Zhu, Ruiqi and Li, Siyuan and Dai, Tianhong and Zhang, Chongjie and Celiktutan, Oya},
  journal={arXiv preprint arXiv:2307.02889},
  year={2023}
}
```