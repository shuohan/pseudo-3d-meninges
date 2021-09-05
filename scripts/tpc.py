#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input')
parser.add_argument('-o', '--output')
args = parser.parse_args()

from nighres.shape import topology_correction

tpc = topology_correction(args.input, 'probability_map')['corrected']
tpc.to_filename(args.output)
