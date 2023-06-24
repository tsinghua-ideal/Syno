import logging
import torch
import torch.nn as nn
import json
import traceback
import os

from KAS import CodeGenOptions, Sampler, MCTS, Path, Placeholder, MockSampler

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel = Placeholder({"H": 128, "W": 128})

    def forward(self, x: torch.Tensor):
        x = self.kernel(x)
        return x

def test_mcts():
    net = Model()
    sampler = Sampler("[H,W]", "[W]", ["H = 128: 1", "W = 128: 1"], [
                      "s_1=2: 2", "k=3: 4"], net=net, depth=5, cuda=False, autoscheduler=CodeGenOptions.Li2018)
    mcts = MCTS(sampler, virtual_loss_constant=1)
    for idx in range(10):
        try:
            receipt, trials = mcts.do_rollout(sampler.root())
            _, path = receipt
            node = trials[0]
            print(f"Iteration {idx}. Sampled {node} for {path}")
            mcts.back_propagate(receipt, 0.5, node[0])
        except Exception as e:
            print(f"Caught error {e}")
            traceback.print_exc()
        if idx in [3, 7]:
            print(f"Garbage collection: size={len(mcts._treenode_store.keys())}->", end="")
            mcts.garbage_collect()
            print(len(mcts._treenode_store.keys()))
    
    for k, v in mcts.virtual_loss_count.items():
        assert v == 0, f"Virtual loss count for {k} is {v}"
    
    # Test serialize
    print("Testing serialization and deserialization. ")
    mcts_serialize = mcts.serialize()
    json.dump(mcts_serialize, open("test_mcts.json", "w"), indent=4)
    mcts_recover = MCTS.deserialize(mcts_serialize, sampler)
    for k, v in mcts_recover._treenode_store.items():
        assert k in mcts._treenode_store, f"Node {k} not in {mcts._treenode_store}"
        assert v == mcts._treenode_store[k], f"Node {k} is {v}, should be {mcts._treenode_store[k]}"
    for k, v in mcts._treenode_store.items():
        if v.N == 0:
            continue
        assert k in mcts_recover._treenode_store, f"Node {k} not in {mcts_recover._treenode_store}"
        assert v == mcts_recover._treenode_store[k], f"Node {k} is {v}, should be {mcts_recover._treenode_store[k]}"
        
    os.remove("test_mcts.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mcts()
