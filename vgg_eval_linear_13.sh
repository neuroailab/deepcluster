DATA="/data5/chengxuz/imagenet_raw"
MODELROOT="${HOME}/deepcluster_models"
MODEL="${MODELROOT}/vgg16/checkpoint.pth.tar"
EXP="${HOME}/deepcluster_exp/vgg_linear_classif_13"

PYTHON="python"

mkdir -p ${EXP}

${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 13 --lr 0.01 \
  --wd -7 --tencrops --verbose --exp ${EXP} --workers 12 --arch vgg16 --batch_size 100
