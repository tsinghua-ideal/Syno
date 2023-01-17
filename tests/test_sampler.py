import torch
from torch import nn
import KAS
from KAS import Sampler, Placeholder

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel_1 = Placeholder({"N": 4, "C": 3, "H": 4, "W": 4})
        self.kernel_2 = Placeholder({"N": 2, "C": 2, "H": 4, "W": 4})

    def forward(self, x: torch.Tensor):
        x = self.kernel_1(x)
        x = torch.einsum("nchw->hw", x)
        x = self.kernel_2(x)
        return x

def test_sampler():
    sampler = Sampler("[H,W]", "[N,C,H,W]", [], ["k_1=3", "s_1=2", "k_2=5", "s_2=4"])

    net = Model()
    path = sampler.sample(net)
    print(net(torch.rand((4, 4))))

if __name__ == "__main__":
    test_sampler()
