#!/usr/bin/env python

import torch
from pytorchviz import make_dot

from ct_synth.network import UNet


x = torch.rand([2, 2, 288, 288]).float().cuda()
net = UNet(2, 2, 5, 32).cuda()
print(net)
dot = make_dot(x, net)
dot.render('network')
