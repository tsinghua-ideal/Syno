import torch
from torch import nn
from typing import Type
import thop
from tqdm.contrib import tzip

mappings = [{'N': 128, 'C_in': 64, 'C_out': 64, 'H': 56}, {'N': 128, 'C_in': 64, 'C_out': 64, 'H': 56}, {'N': 128, 'C_in': 64, 'C_out': 64, 'H': 56}, {'N': 128, 'C_in': 64, 'C_out': 64, 'H': 56}, {'N': 128, 'C_in': 64, 'C_out': 64, 'H': 56}, {'N': 128, 'C_in': 64, 'C_out': 64, 'H': 56}, {'N': 128, 'C_in': 64, 'C_out': 128, 'H': 28}, {'N': 128, 'C_in': 128, 'C_out': 128, 'H': 28}, {'N': 128, 'C_in': 64, 'C_out': 128, 'H': 28}, {'N': 128, 'C_in': 128, 'C_out': 128, 'H': 28}, {'N': 128, 'C_in': 128, 'C_out': 128, 'H': 28}, {'N': 128, 'C_in': 128, 'C_out': 128, 'H': 28}, {'N': 128, 'C_in': 128, 'C_out': 128, 'H': 28}, {'N': 128, 'C_in': 128, 'C_out': 128, 'H': 28}, {'N': 128, 'C_in': 128, 'C_out': 128, 'H': 28}, {'N': 128, 'C_in': 128, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 128, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 256, 'H': 14}, {'N': 128, 'C_in': 256, 'C_out': 512, 'H': 7}, {'N': 128, 'C_in': 512, 'C_out': 512, 'H': 7}, {'N': 128, 'C_in': 256, 'C_out': 512, 'H': 7}, {'N': 128, 'C_in': 512, 'C_out': 512, 'H': 7}, {'N': 128, 'C_in': 512, 'C_out': 512, 'H': 7}, {'N': 128, 'C_in': 512, 'C_out': 512, 'H': 7}, {'N': 128, 'C_in': 512, 'C_out': 512, 'H': 7}]

kernel_sizes = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (1, 1), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (1, 1), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (1, 1), (3, 3), (3, 3), (3, 3), (3, 3)]

k_1=3
k_2=7
s=2
g=32

raw_flops = 133218816
raw_params = 77732
base_flops = 3678454272
base_params = 21335972

kernel2_flops = 829645945
kernel2_params = 4484132

class ConvModule(nn.Module):
    def _cache_sizes(self, x, convs):
        self._sizecache = []

        for conv in convs:
            N, CI, H, W = x.size()
            CO, KH, KW, stride, pad, G = (
                conv.out_channels,
                conv.kernel_size[0],
                conv.kernel_size[1],
                conv.stride[0],
                conv.padding[0],
                conv.groups,
            )
            self._sizecache.append([N, H, W, CO, CI, KH, KW, stride, pad, G])
            x = conv(x)


class Conv(ConvModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias, padding=1
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=padding,
        )

    def forward(self, x):
        return self.conv(x)


class Seq1(ConvModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias, padding=1
    ):
        super(Seq1, self).__init__()
        convs = []
        groups = [2, 4]
        sf = len(groups)
        for i, layer in enumerate(range(sf)):
            g = groups[i]
            convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels // sf,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    padding=padding,
                    groups=g,
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        return torch.cat(outs, dim=1)


class Seq2(ConvModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias, padding=1
    ):
        super(Seq2, self).__init__()
        self.unroll_factor = 16
        g = 2
        self.conv1 = nn.Conv2d(
            in_channels,
            self.unroll_factor,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.convg1 = nn.Conv2d(
            (in_channels - self.unroll_factor),
            (out_channels - self.unroll_factor),
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=g,
        )

    def forward(self, x):
        l_slice = x
        r_slice = x[:, self.unroll_factor :, :, :]

        l_out = self.conv1(l_slice)
        r_out = self.convg1(r_slice)

        return torch.cat((l_out, r_out), 1)


class Seq3(ConvModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, bias, padding=1
    ):
        super(Seq3, self).__init__()
        self.split_factor = 2
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=bias,
                    padding=padding,
                )
                for i in range(self.split_factor)
            ]
        )

    def forward(self, x):
        H = x.shape[2]
        Hg = H // self.split_factor

        outs = []
        for i, conv in enumerate(self.convs):
            x_ = x[:, :, i * Hg : (i + 1) * Hg, :]
            outs.append(conv(x_))

        return torch.cat(outs, 2)

def profile(seq: Type[ConvModule]):
    total_flops = raw_flops
    total_params = raw_params
    for mapping, kernel_size in zip(mappings, kernel_sizes):
        C_in, C_out, H = mapping["C_in"], mapping["C_out"], mapping["H"]
        flops, params = thop.profile( # type: ignore
            model=seq(C_in, C_out, kernel_size, 1, False, kernel_size[0] // 2), 
            inputs=(torch.ones(1, C_in, H, H),), 
            verbose=False
        )
        total_flops += flops
        total_params += params
    total_flops = int(total_flops)
    total_params = int(total_params)
    return total_flops, total_params

assert profile(Conv) == (base_flops, base_params)
for seq in [Seq1, Seq2, Seq3]:
    flops, params = profile(seq)
    print(f"Over {seq.__name__}: FLOPs speedup = {flops / kernel2_flops}, params = {params / kernel2_params}")
    