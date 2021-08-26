#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-1', '--t1w', help='Input T1w image.')
parser.add_argument('-2', '--t2w', help='Input T2w image.')
parser.add_argument('-o', '--output-dir', help='Output directory.')
parser.add_argument('-c', '--checkpoint', help='Checkpoint of model parameters.')
parser.add_argument('-b', '--batch-size', type=int, default=2,
                    help='Testing batch size.')
parser.add_argument('-f', '--num-channels', type=int, default=64)
parser.add_argument('-u', '--use-cuda', action='store_true')
parser.add_argument('-T', '--target-shape', type=int, default=(288, 288),
                    nargs=2, help='Pad or crop the slices to this shape.')
parser.add_argument('-C', '--combine', choices={'median', 'mean'},
                    default='median')
parser.add_argument('-O', '--output-data-mode', default='ct_mask',
                    choices={'ct_mask', 'ct', 'mask'})
args = parser.parse_args()

from ct_synth.test import TesterBoth, TesterCT, TesterMask


if args.t1w and args.t2w:
    args.input_data_mode = 't1w_t2w'
elif args.t1w:
    args.input_data_mode = 't1w'
elif args.t2w:
    args.input_data_mode = 't2w'

if args.output_data_mode == 'ct_mask':
    TesterBoth(args).test()
elif args.output_data_mode == 'ct':
    TesterCT(args).test()
elif args.output_data_mode == 'mask':
    TesterMask(args).test()
