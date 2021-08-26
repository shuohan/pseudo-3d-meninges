#!/usr/bin/env python

import torch
from pytorchviz import make_dot

from deep_meninges.network import OutBlock


x = torch.rand([3, 8, 128, 128]).float().cuda()
block = OutBlock(8).cuda()
print(block)
dot = make_dot(x, block)
dot.render('out_block')
