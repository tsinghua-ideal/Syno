import logging
import torch
from torch import nn
import KAS
from KAS import Sampler, Placeholder, CodeGenOptions

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel_1 = Placeholder({"N": 16, "C": 16, "H": 16, "W": 16})
        # self.kernel_2 = Placeholder({"N": 4, "C": 3, "H": 16, "W": 16})

    def forward(self, x: torch.Tensor):
        x = self.kernel_1(x)
        # x = torch.einsum("nchw->hw", x)
        # x = self.kernel_2(x)
        return x

def test_sampler():
    sampler = Sampler("[H,W]", "[N,C,H,W]", [], ["s_1=2", "s_2=2"], depth=2, autoscheduler=CodeGenOptions.AutoScheduler.ComputeRoot)

    net = Model()
    path = sampler.sample(net)
    print(net(torch.rand((16, 16))))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_sampler()
