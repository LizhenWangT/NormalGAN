!/usr/bin/env bash

#expm=`basename $1`
modelname='pretrained'
csvname=$1
savename=$2
modeldir=`realpath model/$modelname`
savedir=`realpath results/$savename`
echo "modeldir: "$modeldir
echo "savedir: "$savedir
mkdir -p $savedir

cd src
CUDA_VISIBLE_DEVICES=0 unbuffer python3 test_offline.py \
    --modeldir=$modeldir \
    --savedir=$savedir \
    --dataset_dir=../datasets \
    --depth_net_name=depth_net_80000.pth \
    --back_net_name=back_net_80000.pth \
    --f_alb_net_name=f_alb_net_80000.pth \
    --b_alb_net_name=b_alb_net_80000.pth \
    --index_file=$csvname \
    --low_thres=1000 \
    --up_thres=2300 \
    --batch_size=1
