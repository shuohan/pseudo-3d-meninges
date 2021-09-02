#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Calculate SDF from mask.')
parser.add_argument('-i', '--input-dir')
parser.add_argument('-o', '--output-dir')
parser.add_argument('-m', '--max-distance', default=4, type=float)
parser.add_argument('-n', '--num-workers', default=1, type=int)
args = parser.parse_args()


import nibabel as nib
import numpy as np
from skfmm import distance
from pathlib import Path
from multiprocessing import Pool


def calc_sdf(mask_fn):
    name = mask_fn.name.replace('mask', 'sdf')
    filename = Path(args.output_dir, name)
    print(mask_fn, filename)
    mask_obj = nib.load(mask_fn)
    mask = mask_obj.get_fdata()
    dist = -distance(mask - 0.5)
    dist[dist > args.max_distance] = args.max_distance
    dist[dist < -args.max_distance] = -args.max_distance
    output_obj = nib.Nifti1Image(dist, mask_obj.affine, mask_obj.header)
    output_obj.to_filename(filename)


Path(args.output_dir).mkdir(parents=True, exist_ok=True)
mask_filenames = list(Path(args.input_dir).rglob('*mask*'))
with Pool(args.num_workers) as pool:
    pool.map(calc_sdf, mask_filenames)
