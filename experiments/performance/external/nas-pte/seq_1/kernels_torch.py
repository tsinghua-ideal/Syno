import torch

groups = [2, 4]

class kernel_generated(torch.nn.Module):
    def __init__(self, C_in: int, C_out: int, H: int, k: int):
        super().__init__()
        split_factor = len(groups)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(C_in, C_out // split_factor, (k, k), groups=g)
            for g in groups
        ])

    def forward(self, x):
        results = [conv(x) for conv in self.convs]
        return torch.cat(results, dim=1)
