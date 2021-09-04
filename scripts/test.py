#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--images', nargs='+',
    help=('Input images in the order of the input data mode in training '
          'configurations.')
)
parser.add_argument('-d', '--dirname', help='Input directory.')
parser.add_argument('-o', '--output-dir', help='Output directory.')
parser.add_argument('-c', '--checkpoint', help='Checkpoint of model parameters.')
parser.add_argument('-a', '--train-config', help='Configurations from training.')
parser.add_argument('-b', '--batch-size', type=int, default=8,
                    help='Testing batch size.')
parser.add_argument('-u', '--use-cuda', action='store_true')
parser.add_argument('-T', '--target-shape', type=int, default=(288, 288),
                    nargs=2, help='Pad or crop the slices to this shape.')
parser.add_argument('-C', '--combine', choices={'median', 'mean'},
                    default='mean')
parser.add_argument('-M', '--max-sdf-value', default=4, type=float)
parser.add_argument('-w', '--num-workers', default=0, type=int)
args = parser.parse_args()

from deep_meninges.test import Tester, TesterDataset


if args.dirname is not None:
    TesterDataset(args).test()
else:
    Tester(args).test()
