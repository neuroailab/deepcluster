# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/data5/chengxuz/imagenet_raw/train"
ARCH="resnet50_dc_np"
LR=0.1
WD=-5
K=10000
WORKERS=12
EXP="/data/${USER}/dc_test/exp_50_no_pool"
PYTHON="python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} \
  --epochs 500 --batch 160 --seed 29 --lr 0.03
