#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='Calculate SDF from mask.')
parser.add_argument('-i', '--input-dir')
parser.add_argument('-o', '--output-dir')
parser.add_argument('-T', '--threshold', default=0, type=float)
parser.add_argument('-s', '--source-name', default='sdf-orig')
parser.add_argument('-t', '--target-name', default='mask')
parser.add_argument('-n', '--num-workers', default=1, type=int)
args = parser.parse_args()


import nibabel as nib
import numpy as np
from pathlib import Path
from multiprocessing import Pool


def calc_mask(sdf_fn):
    name = sdf_fn.name.replace(args.source_name, args.target_name)
    filename = Path(args.output_dir, name)
    print(sdf_fn, filename)
    sdf_obj = nib.load(sdf_fn)
    sdf = sdf_obj.get_fdata()
    mask = sdf <= args.threshold
    output_obj = nib.Nifti1Image(mask, sdf_obj.affine, sdf_obj.header)
    output_obj.to_filename(filename)


Path(args.output_dir).mkdir(parents=True, exist_ok=True)
sdf_filenames = list(Path(args.input_dir).rglob('*' + args.source_name + '*'))
with Pool(args.num_workers) as pool:
    pool.map(calc_mask, sdf_filenames)
