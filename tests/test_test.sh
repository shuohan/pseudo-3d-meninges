#!/usr/bin/env bash

export PYTHONPATH=~/Code/shuo/deep-networks/mallika-ct-synth:~/Code/shuo/deep-networks/ptxl:~/Code/shuo/deep-networks/pytorch-unet:~/Code/shuo/utils/resize
export CUDA_VISIBLE_DEVICES=1

t1w=../data_test/MICA-A38_t1w.nii.gz 
t2w=../data_test/MICA-A38_t2w.nii.gz 
output_dir=results_test_t1w_t2w
checkpoint=results_train_valid_t1w_t2w/checkpoints/epoch-1.pt

# ../scripts/test.py -1 $t1w -2 $t2w -o $output_dir -c $checkpoint -u -f 2

output_dir=results_test_t1w
checkpoint=results_train_valid_t1w/checkpoints/epoch-1.pt
../scripts/test.py -1 $t1w -o $output_dir -c $checkpoint -u -f 2
