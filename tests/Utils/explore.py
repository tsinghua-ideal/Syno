import torch
from torch import nn, Tensor

# KAS
from KAS import Sampler, TreePath, Placeholder, Explorer
from KAS.Bindings import CodeGenOptions

class KASGrayConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Placeholder({'N': 5000, 'C_out': 32, 'H': 28, 'W': 28}),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
        )
        self.linear = nn.Linear(32*7*7, 10)

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)

        x = image.squeeze(1)
        for layer in self.layers:
            x = layer(x)

        return self.linear(x.view(B, -1))

model = KASGrayConv().cuda()
sampler = Sampler(
    input_shape="[N,H,W]",
    output_shape="[N,C_out,H,W]",
    primary_specs=["N=4096: 1", "H=256", "W=256", "C_out=100"],
    coefficient_specs=["s_1=2", "k_1=3", "k_2=5"],
    seed=0xdeadbeaf,
    depth=8,
    dim_lower=2,
    dim_upper=8,
    save_path='./saves',
    cuda=torch.cuda.is_available(),
    net=model,
    fixed_io_pairs=[(0, 0)],
    autoscheduler=CodeGenOptions.AutoScheduler.Anderson2021
)

explorer = Explorer(sampler)
explorer.interactive()
