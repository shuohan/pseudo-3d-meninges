#!/usr/bin/env python

import torch
from pytorchviz import make_dot

from deep_meninges.network import UNet


x = torch.rand([2, 2, 32, 32]).float().cuda()
net = UNet(2, 2, 3, 8).cuda()
print(net)
dot = make_dot(x, net)
dot.render('network')
