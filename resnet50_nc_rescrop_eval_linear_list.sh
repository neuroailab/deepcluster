DATA="/data5/chengxuz/Dataset/imagenet_raw"
MODELROOT="/data/${USER}/dc_test"
MODEL="${MODELROOT}/exp_50_no_class/checkpoint.pth.tar"
EXP="${HOME}/deepcluster_exp/resnet50_nc_rescrop_linear_classif_list"

PYTHON="python"

mkdir -p ${EXP}

${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 105,106,107,108 --lr 0.01 \
  --pool "4,4,0;2,2,0;2,2,0;1,1,0" --linear 50176,100352,50176,100352 \
  --wd -7 --verbose --exp ${EXP} --workers 12 --arch resnet \
  --batch_size 100 --lr_boundary
