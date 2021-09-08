#!/usr/bin/env bash

export PYTHONPATH=~/Code/shuo/deep-networks/pseudo-3d-meninges:~/Code/shuo/deep-networks/ptxl:~/Code/shuo/deep-networks/pytorch-unet:~/Code/shuo/utils/resize
export CUDA_VISIBLE_DEVICES=1

t1w=data/OAS30002-d0653_t1w.nii 
t2w=data/OAS30002-d0653_t2w.nii 
output_dir=results_test_7cffbce
checkpoint=checkpoint_7cffbce.pt

# ../scripts/test.py -i $t1w $t2w -o $output_dir -c $checkpoint -u \
#     -a config.json -w 4
../scripts/test.py -d data -o $output_dir -c $checkpoint -u \
    -a config_7cffbce.json -w 4
