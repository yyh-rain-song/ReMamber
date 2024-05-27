import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

_C.TYPE = ''
# Model name
_C.NAME = '_tiny_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.RESUME = ''
# Number of classes, overwritten in data preparation
_C.NUM_CLASSES = 1000
# Dropout rate
_C.DROP_RATE = 0.0
# Drop path rate
_C.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.LABEL_SMOOTHING = 0.1

_C.USE_CHECKPOINT = False

# MMpretrain models for test
_C.MMCKPT = False

_C.PATCH_SIZE = 4
_C.IN_CHANS = 3
_C.DEPTHS = [2, 2, 9, 2]
_C.EMBED_DIM = 96
_C.SSM_D_STATE = 16
_C.SSM_RATIO = 2.0
_C.SSM_RANK_RATIO = 2.0
_C.SSM_DT_RANK = "auto"
_C.SSM_ACT_LAYER = "silu"
_C.SSM_CONV = 3
_C.SSM_CONV_BIAS = True
_C.SSM_DROP_RATE = 0.0
_C.SSM_INIT = "v0"
_C.SSM_FORWARDTYPE = "v4"
_C.MLP_RATIO = 0.0
_C.MLP_ACT_LAYER = "gelu"
_C.MLP_DROP_RATE = 0.0
_C.PATCH_NORM = True
_C.NORM_LAYER = "ln2d"
_C.DOWNSAMPLE = "v1"
_C.PATCHEMBED = "v1"
_C.GMLP = False