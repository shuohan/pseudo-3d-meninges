#!/usr/bin/env bash

export PYTHONPATH=~/Code/shuo/deep-networks/mallika-ct-synth:~/Code/shuo/deep-networks/ptxl:~/Code/shuo/deep-networks/pytorch-unet:~/Code/shuo/utils/resize
export CUDA_VISIBLE_DEVICES=1
data=../tmp
data_valid=../data_valid

../scripts/train.py -t $data -v $data_valid -o results_train_valid_t1w \
    -w 1 -f 2 -s 1 -V 1 -C 1 -e 1 -I t1w
