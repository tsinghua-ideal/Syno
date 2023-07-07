import logging
import torch
import torch.nn as nn
import json
import traceback
import os
from random import random

from KAS import CodeGenOptions, Sampler, MCTS, Placeholder, TreeExplorer

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel = Placeholder({"H": 128, "W": 128})

    def forward(self, x: torch.Tensor):
        x = self.kernel(x)
        return x

def test_tree_explorer():
    net = Model()
    sampler = Sampler("[H,W]", "[H,W]", ["H:2", "W:2"], [
                      "s=3:2", "k=4:4"], net=net, depth=5, cuda=False, autoscheduler=CodeGenOptions.Li2018)
    mcts = MCTS(sampler, virtual_loss_constant=1)
    for idx in range(30):
        try:
            receipt, trials = mcts.do_rollout(sampler.root())
            _, path = receipt
            node = trials[0]
            mcts.back_propagate(receipt, random(), node[0])
        except Exception as e:
            print(f"Caught error {e}")
            traceback.print_exc()
        if idx in [3, 7]:
            mcts.garbage_collect()
    
    for k, v in mcts.virtual_loss_count.items():
        assert v == 0, f"Virtual loss count for {k} is {v}"
    
    json.dump(mcts.serialize(), open("test_mcts.json", "w"), indent=4)
    
    # Test begin. Now we have a serialized mcts saved in test_mcts.json
    # Replace test_mcts.json with your own serialized mcts
    mcts_serialize = json.load(open("test_mcts.json", "r"))
    mcts_recover = MCTS.deserialize(mcts_serialize, sampler)
    
    explorer = TreeExplorer(mcts_recover)
    try:
        explorer.interactive()
    except KeyboardInterrupt:
        pass
        
    os.remove("test_mcts.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_tree_explorer()
