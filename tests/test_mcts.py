import logging
import torch
import torch.nn as nn

from KAS import CodeGenOptions, Sampler, MCTS, Path, Placeholder


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel = Placeholder({"W": 128})

    def forward(self, x: torch.Tensor):
        x = self.kernel(x)
        return x

def test_mcts():
    net = Model()
    sampler = Sampler("[H,W]", "[H,W]", ["H = 128: 1", "W: 3"], ["s_1=2: 2", "k_1=3", "4"], net=net, depth=5, cuda=False, autoscheduler=CodeGenOptions.Li2018)
    mcts = MCTS(sampler)
    in_tensor = torch.randn((128, 128))
    for idx in range(10):
        receipt, node = mcts.do_rollout(sampler.root())
        _, path = receipt
        print(f"Iteration {idx}. Sampled {node.path} for {path}:")
        print(sampler.path_to_strs(path))
        for i in range(len(path)):
            child = sampler.visit(Path(path.abs_path[:i]))
            print(f"Node {child} has children:", child.get_children_types())
        kernel_packs, _ = sampler.realize(net, node, f"test_mcts_{idx}")
        sampler.replace(net, kernel_packs)
        print(f"Computing forward {idx}...")
        print(f"Result: {net(in_tensor)}")
        mcts.back_propagate(receipt, 1.0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_mcts()
