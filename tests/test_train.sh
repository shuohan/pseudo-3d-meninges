#!/usr/bin/env bash

export PYTHONPATH=~/Code/shuo/deep-networks/pseudo-3d-meninges:~/Code/shuo/deep-networks/ptxl:~/Code/shuo/deep-networks/pytorch-unet:
export CUDA_VISIBLE_DEVICES=1
data=data
data_valid=data
output=results_train_valid

../scripts/train.py -t $data -o $output \
    -w 1 -D 3 -f 2 -s 1 -V 1 -C 1 -e 1 \
    -O outer-mask_outer-sdf_inner-mask_inner-sdf
