import torch

unroll_factor = 16
unrollconv_groups = 2

class kernel_generated(torch.nn.Module):
    def __init__(self, C_in: int, C_out: int, H: int, k: int):
        super().__init__()
        self.conv_left = torch.nn.Conv2d(C_in, unroll_factor, (k, k))
        self.conv_right = torch.nn.Conv2d(C_in - unroll_factor, C_out - unroll_factor, (k, k), groups=unrollconv_groups)

    def forward(self, x):
        _, right = torch.split(x, [unroll_factor, x.shape[1] - unroll_factor], dim=1)
        left = self.conv_left(x)
        right = self.conv_right(right)
        return torch.cat([left, right], dim=1)
