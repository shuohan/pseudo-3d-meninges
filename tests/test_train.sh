#!/usr/bin/env bash

export PYTHONPATH=~/Code/shuo/deep-networks/pseudo-3d-meninges:~/Code/shuo/deep-networks/ptxl:~/Code/shuo/deep-networks/pytorch-unet:
export CUDA_VISIBLE_DEVICES=1
data=data_multi
data_valid=data_multi
output=results_train_valid

../scripts/train.py -t $data -v $data_valid -o $output \
    -w 0 -D 3 -f 2 -s 1 -V 1 -C 2 -e 2 \
    -O outer-mask_outer-sdf_inner-mask_inner-sdf \
    -n 16 -N 32
