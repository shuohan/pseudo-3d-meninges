#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train-dir', help='Directory of training dataset.')
parser.add_argument('-v', '--valid-dir', default='',
                    help='Directory of validation dataset.')
parser.add_argument('-o', '--output-dir', help='Directory of output results.')
parser.add_argument('-b', '--batch-size', type=int, default=8,
                    help='Training batch size.')
parser.add_argument('-B', '--test-batch-size', type=int, default=1,
                    help='Testing batch size.')
parser.add_argument('-w', '--num-workers', type=int, default=2,
                    help='Number of DataLoader multiprocessing workers.')
parser.add_argument('-f', '--num-channels', type=int, default=64,
                    help='The number of network features of the first block.')
parser.add_argument('-e', '--num-epochs', type=int, default=60,
                    help='Number of training epochs')
parser.add_argument('-l', '--learning-rate', type=float, default=0.0002,
                    help='Learning rate for the generator.')
parser.add_argument('-1', '--lambda-ct', type=float, default=2,
                    help='Weighting for CT synthesis.')
parser.add_argument('-2', '--lambda-mask', type=float, default=100,
                    help='Weighting for intracranial segmentation.')
parser.add_argument('-T', '--target-shape', type=int, default=(288, 288),
                    nargs=2, help='Pad or crop the slices to this shape.')
parser.add_argument('-c', '--checkpoint')
parser.add_argument('-C', '--checkpoint-save-step', type=int,
                    default=float('inf'))
parser.add_argument('-s', '--valid-step', type=int, default=float('inf'))
parser.add_argument('-S', '--image-save-step', default=1, type=int)
parser.add_argument('-V', '--valid-save-step', default=float('inf'), type=int)
parser.add_argument('-z', '--image-save-zoom', default=1, type=int)
parser.add_argument('-I', '--input-data-mode', default='t1w_t2w',
                    choices={'t1w_t2w', 't1w', 't2w'})
parser.add_argument('-O', '--output-data-mode', default='ct_mask',
                    choices={'ct_mask', 'ct', 'mask'})
parser.add_argument('-m', '--memmap', action='store_true')
parser.add_argument('-d', '--scale-aug', type=float, default=1.2)
parser.add_argument('-F', '--finetune', action='store_true')
args = parser.parse_args()


from ct_synth.train import Trainer, TrainerValid


trainer = TrainerValid(args) if args.valid_dir else Trainer(args)
trainer.train()
