DATA="/data5/chengxuz/Dataset/imagenet_raw"
MODELROOT="/data/${USER}/dc_test/exp"
MODEL="${MODELROOT}/checkpoint.pth.tar"
EXP="${MODELROOT}/resnet_exp_lr_bd_bwd"

PYTHON="python"

mkdir -p ${EXP}

${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 113 --lr 0.01 \
  --wd -4 --verbose --exp ${EXP} --workers 12 --arch resnet18_dc \
  --batch_size 256 --lr_boundary
