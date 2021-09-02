#!/usr/bin/env python

import torch
from pytorchviz import make_dot

from deep_meninges.network import UNet


x = torch.rand([2, 2, 32, 32]).float().cuda()
output_attrs = {
    'outer': ['mask', 'sdf'],
    'inner': ['mask', 'sdf'],
}

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = UNet(2, output_attrs, 3, 8, 1).cuda()
    def forward(self, x):
        out = self.net(x)
        return out['outer'] + out['inner']

net = Network()
print(net)
# print(net)
# y = net(x)

dot = make_dot(x, net)
dot.render('network')
