DATA="/data5/chengxuz/Dataset/imagenet_raw"
MODELROOT="/data/chengxuz/dc_test"
MODEL="${MODELROOT}/vgg16/checkpoint.pth.tar"
EXP="${HOME}/deepcluster_exp/vgg_rescrop_linear_classif_list"

PYTHON="python"

mkdir -p ${EXP}

${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 4,7,10,13 --lr 0.01 \
  --pool "8,8,0;8,8,0;4,4,0;2,2,0" --linear 25088,12544,25088,25088 \
  --wd -7 --verbose --exp ${EXP} --workers 12 --arch vgg16 \
  --batch_size 100 --lr_boundary --res_crop
