import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import KAS
from KAS import Sampler, Placeholder, CodeGenOptions

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel_1 = Placeholder({"N": 16, "C": 16, "H": 16, "W": 16})
        self.kernel_2 = Placeholder({"N": 4, "C": 3, "H": 16, "W": 16})

    def forward(self, x: torch.Tensor):
        x = self.kernel_1(x)
        x = torch.einsum("nchw->hw", x)
        x = self.kernel_2(x)
        return x

def test_sampler():
    sampler = Sampler("[H,W]", "[N,C,H,W]", [], ["s_1=2", "s_2=2"], depth=2, autoscheduler=CodeGenOptions.AutoScheduler.ComputeRoot)

    net = Model()
    path = sampler.sample(net)
    in_tensor = torch.rand((16, 16))
    out_tensor = net(in_tensor)
    print("First output: ")
    print(out_tensor)
    target = torch.empty((4, 3, 16, 16))
    # compute gradient
    loss = F.mse_loss(out_tensor, target)
    print("First loss:", loss)
    loss.backward()
    # descent
    with torch.no_grad():
        for param in net.parameters():
            param -= param.grad * 0.01
            param.grad.zero_()
    out_tensor = net(in_tensor)
    print("Second output: ")
    print(out_tensor)
    new_loss = F.mse_loss(out_tensor, target)
    print("Second loss:", new_loss)
    assert new_loss < loss

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_sampler()
