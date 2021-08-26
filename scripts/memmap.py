#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-dir')
parser.add_argument('-o', '--output-dir')
args = parser.parse_args()

import re
import nibabel as nib
import numpy as np
from pathlib import Path


dtype_str = 'float32'
dtype = getattr(np, dtype_str)
Path(args.output_dir).mkdir(exist_ok=True, parents=True)

for fn in Path(args.input_dir).glob('*.nii*'):
    print(fn)
    data = nib.load(fn).get_fdata(dtype=dtype)
    outfn = re.sub(r'\.nii\.gz$', '_data.dat', fn.name)
    outfn = Path(args.output_dir, outfn)
    fp = np.memmap(outfn, dtype=dtype_str, mode='w+', shape=data.shape)
    fp[:] = data[:]
    fp.flush()

    shape_outfn = re.sub(r'_data\.dat$', '_shape.npy', str(outfn))
    np.save(shape_outfn, data.shape)

    dtype_outfn = re.sub(r'_data\.dat$', '_dtype.txt', str(outfn))
    with open(dtype_outfn, 'w') as f:
        f.write(dtype_str)
