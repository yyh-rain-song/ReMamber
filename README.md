<div align="center">
<h1>ReMamber </h1>
<h3>ReMamber: Referring Image Segmentation with Mamba Twister</h3>

Yuhuan Yang*<sup>1</sup>, Chaofan Ma*<sup>1</sup>, Jiangchao Yao<sup>1</sup>, Zhun Zhong<sup>2</sup>, Ya Zhang<sup>1</sup>, Yanfeng Wang<sup>1</sup>

<sup>1</sup>  Shanghai Jiao Tong University, <sup>2</sup>  University of Nottingham

Paper: ([arXiv 2403.17839](https://arxiv.org/abs/2403.17839))

<div align="left">

## Overview
<p align="center">
  <img src="assets/teaser.png" alt="accuracy" width="90%">
</p>

## Preparation
#### Prepare environment
- python 3.10.13: 
  - ``conda create -n remamber python=3.10.13``
- torch 2.1.1 + cu118: 
  - ``pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118``
- install dependencies:
  - ``pip install -r requirements.txt``
- build kernel for VMamba dependencies: 
  - ``cd selective_scan && pip install .``
- pretrain checkpoint
  - download from [VMamba](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_base_0229_ckpt_epoch_237.pth)
  - Create a ``pretrain/`` folder and place the checkpoint in it

#### Prepare dataset
Follow [ref_dataset/prepare_dataset.md](ref_dataset/prepare_dataset.md) to prepare dataset.

## Training
This implementation only supports multi-gpu, DistributedDataParallel training. To train ReMamber using 8 GPUs, run:
``` bash
python -m torch.distributed.launch --nproc_per_node=8 \
                                   --use_env main.py \
                                   --model ReMamber_Conv \
                                   --output_dir your/logging/directory \
                                   --if_amp \
                                   --batch_size 8 \
                                   --model-ema
```

*Todo: We will release the checkpoint soon.

## Citation
If this code is useful for your research, please consider citing:
```
@article{yang2024remamber,
  title   = {ReMamber: Referring Image Segmentation with Mamba Twister},
  author  = {Yuhuan Yang and Chaofan Ma and Jiangchao Yao and Zhun Zhong and Ya Zhang and Yanfeng Wang},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2403.17839}
}
```
## Acknowledgements
This project is based on [VMamba](https://github.com/MzeroMiko/VMamba), [Vim](https://github.com/hustvl/Vim), [LAVT](https://github.com/yz93/LAVT-RIS). Thanks for their wonderful works.
