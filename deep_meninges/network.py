import torch
import torch.nn.functional as F

from pytorch_unet.blocks import _ConvBlock, _ContractingBlock
from pytorch_unet.blocks import _ExpandingBlock, _TransUpBlock
from pytorch_unet.unet import _UNet
from resize.pytorch import ResizeTorch

from kornia.filters.kernels import get_spatial_gradient_kernel2d
from kornia.filters.kernels import normalize_kernel2d
from kornia.filters.kernels import get_gaussian_kernel2d


class Resize(torch.nn.Module):
    def __init__(self, dxyz, order=1):
        super().__init__()
        self.dxyz = (dxyz, ) * 2
        self.order = order
        self._resizer = None

    def forward(self, x):
        if self._resizer is None:
            self._resizer = ResizeTorch(x, self.dxyz, order=self.order)
            self._resizer.resize()
        else:
            recompute = self._resizer.image.shape != x.shape
            self._resizer.image = x
            if recompute:
                self._resizer._calc_sampling_coords()
                self._resizer._format_coords()
            self._resizer._resize()
        return self._resizer.result

    def extra_repr(self):
        return f'dxyz={self.dxyz}, order={self.order}'


class ConvBlock(_ConvBlock):
    def _create_conv(self):
        return torch.nn.Conv2d(self.in_channels, self.out_channels, 3,
                               padding=1, bias=True)
    def _create_activ(self):
        return torch.nn.LeakyReLU(0.01)

    def _create_norm(self):
        return torch.nn.BatchNorm2d(self.out_channels)

    def _create_dropout(self):
        return torch.nn.Dropout2d(0.2)


class ProjBlock(ConvBlock):
    def _create_conv(self):
        return torch.nn.Conv2d(self.in_channels, self.out_channels, 1)


class ContractingBlock(_ContractingBlock):
    def _create_conv0(self):
        return ConvBlock(self.in_channels, self.mid_channels)

    def _create_conv1(self):
        return ConvBlock(self.mid_channels, self.out_channels)


class ExpandingBlock(_ExpandingBlock):
    def _create_conv0(self):
        in_channels = self.in_channels + self.shortcut_channels
        return ConvBlock(in_channels, self.out_channels)

    def _create_conv1(self):
        return ConvBlock(self.out_channels, self.out_channels)


class TransUpBlock(_TransUpBlock):
    def _create_conv(self):
        return ProjBlock(self.in_channels, self.out_channels)

    def _create_up(self):
        return Resize(0.5, 1)


class OutBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, sigma=3, gauss_size=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._init_gauss(sigma, gauss_size)
        self._init_sobel()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(self.in_channels, 1, 1)
            for _ in range(self.out_channels)
        ])

    def _init_gauss(self, sigma, size):
        kernel = get_gaussian_kernel2d((size, size), (sigma, sigma))
        kernel = normalize_kernel2d(kernel)[None, None]
        self.register_buffer('gauss', kernel)
        self._gauss_padding = (size // 2, ) * 2

    def _init_sobel(self):
        kernel = get_spatial_gradient_kernel2d('sobel', 1)
        kernel = normalize_kernel2d(kernel).unsqueeze(1)
        self.register_buffer('sobel', kernel)
        self._sobel_padding = (1, 1)

    def forward(self, x):
        mask = self.convs[0](x)
        mask = torch.sigmoid(mask)
        levelset = self.convs[1](x)
        others = [conv(x) for conv in self.convs[2:]]

        with torch.no_grad():
            mask_pad = F.pad(mask, self._gauss_padding * 2, mode='replicate')
            blur = F.conv2d(mask_pad, self.gauss)
            blur = F.pad(blur, self._sobel_padding * 2, mode='replicate')
            edge = F.conv2d(blur, self.sobel)
            edge = torch.sqrt(torch.sum(edge * edge, dim=1, keepdim=True))

        return mask, levelset, edge, *others


class OutBlocks(torch.nn.ModuleDict):
    def __init__(self, in_channels, output_attrs, num_hidden=0):
        super().__init__()
        self.in_channels = in_channels
        self.output_attrs = output_attrs
        self.num_hidden = num_hidden

        for k, v in self.output_attrs.items():
            self[k] = torch.nn.Sequential(
                *[
                    ConvBlock(in_channels, in_channels)
                    for _ in range(self.num_hidden)
                ],
                OutBlock(in_channels, len(v))
            )

    def forward(self, x):
        return {k: block(x) for k, block in self.items()}


class UNet(_UNet):
    """The UNet.

    """
    def __init__(self, in_channels, output_attrs, num_trans_down,
                 first_channels, num_out_hidden=0, max_channels=1024):
        self.output_attrs = output_attrs
        self.num_out_hidden = num_out_hidden
        super().__init__(in_channels, 1, num_trans_down,
                         first_channels, max_channels)

    def _create_ib(self, in_channels, out_channels, mid_channels):
        return ContractingBlock(in_channels, out_channels, mid_channels)

    def _create_cb(self, in_channels, out_channels, mid_channels):
        return ContractingBlock(in_channels, out_channels, mid_channels)

    def _create_td(self):
        return Resize(2, 1)

    def _create_tu(self, in_channels, out_channels):
        return TransUpBlock(in_channels, out_channels)

    def _create_eb(self, in_channels, shortcut_channels, out_channels):
        return ExpandingBlock(in_channels, shortcut_channels, out_channels)

    def _create_out(self, in_channels):
        return OutBlocks(in_channels, self.output_attrs, self.num_out_hidden)
