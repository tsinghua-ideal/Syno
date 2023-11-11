import logging
import torch
import torch.nn as nn
import os, sys
from random import random

from KAS import CodeGenOptions, Sampler, Placeholder

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from search.mcts import MCTSTree, MCTSExplorer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel = Placeholder({"H": 128, "W": 128})

    def forward(self, x: torch.Tensor):
        x = self.kernel(x)
        return x


def test_tree_explorer():
    net = Model()
    sampler = Sampler(
        "[H,W]",
        "[H,W]",
        ["H:2", "W:2"],
        ["s=3:2", "k=4:4"],
        net=net,
        depth=5,
        cuda=False,
        autoscheduler=CodeGenOptions.Li2018,
    )
    mcts = MCTSTree(sampler, virtual_loss_constant=1, simulate_retry_period=1e6)
    for idx in range(10):
        receipts, trials = [], []
        for _ in range(2):
            receipt, trial = mcts.do_rollout()
            receipts.append(receipt)
            trials.append(trial)
        print(f"Iteration {idx}. Sampled {receipts}. ")
        for receipt, trial in zip(receipts, trials):
            mcts.back_propagate(receipt, random(), trial[0][0])

    explorer = MCTSExplorer(net, sampler, mcts)
    try:
        explorer.interactive()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_tree_explorer()
