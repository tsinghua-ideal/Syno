import torch

split_factor = 2

class kernel_generated(torch.nn.Module):
    def __init__(self, C_in: int, C_out: int, H: int, k: int):
        super().__init__()
        self.H = H
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(C_in, C_out, (k, k))
        ])

    def forward(self, x):
        sizes = [self.H // split_factor] * split_factor
        if self.H % split_factor != 0:
            sizes[-1] += self.H % split_factor
        xs = torch.split(x, sizes, dim=2)
        results = [conv(xs[i]) for i, conv in enumerate(self.convs)]
        return torch.cat(results, dim=2)
