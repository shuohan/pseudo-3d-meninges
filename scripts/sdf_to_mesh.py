#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
args = parser.parse_args()


from nighres.surface import levelset_to_mesh
from pathlib import Path


output_dir = Path(args.output).parent
output_dir.mkdir(exist_ok=True, parents=True)
file_name = Path(args.output).name
mesh = levelset_to_mesh(args.input, save_data=True, overwrite=True,
                        output_dir=str(output_dir), file_name=file_name)
