import logging
import torch
import torch.nn as nn
import json
import traceback

from KAS import CodeGenOptions, Sampler, MCTS, Path, Placeholder


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel = Placeholder({"H": 128, "W": 128})

    def forward(self, x: torch.Tensor):
        x = self.kernel(x)
        return x


def test_mcts():
    net = Model()
    sampler = Sampler("[H,W]", "[H,W]", ["H = 128: 1", "W = 128: 1"], [
                      "s_1=2: 2", "k=3: 2", "4"], net=net, depth=5, cuda=False, autoscheduler=CodeGenOptions.Li2018)
    mcts = MCTS(sampler)
    for idx in range(2):
        try:
            receipt, trials = mcts.do_rollout(sampler.root())
            _, path = receipt
            node = trials[0]
            print(f"Iteration {idx}. Sampled {node} for {path}:")
            mcts.back_propagate(receipt, 1.0)
        except Exception as e:
            print(f"Caught error {e}")
            traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_mcts()
