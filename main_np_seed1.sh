# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/data5/chengxuz/Dataset/imagenet_raw/train"
ARCH="resnet18_dc_np"
LR=0.1
WD=-5
K=10000
WORKERS=12
EXP="/mnt/fs4/chengxuz/deepcluster_models/res18_np_seed1"
PYTHON="python"

mkdir -p ${EXP}

${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} \
  --epochs 500 --seed 1
