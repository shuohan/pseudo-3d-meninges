#!/usr/bin/env python

import torch
from pytorchviz import make_dot
import nibabel as nib
import numpy as np

from deep_meninges.network import OutBlock


x = nib.load('MICA-A02_00_TIV_MASK.nii.gz').get_fdata()
x = x[:, :, 128][None, None]
x = torch.tensor(x, dtype=torch.float32).cuda()
# x = torch.rand([3, 1, 128, 128]).float().cuda()
block = OutBlock(1, ['mask', 'sdf']).cuda()
mask, levelset, edge = block(x)

nib.Nifti1Image(edge.cpu().detach().numpy().squeeze(), np.eye(4)).to_filename('edge.nii.gz')
nib.Nifti1Image(mask.cpu().detach().numpy().squeeze(), np.eye(4)).to_filename('mask.nii.gz')
print(block)
dot = make_dot(x, block)
dot.render('out_block')
