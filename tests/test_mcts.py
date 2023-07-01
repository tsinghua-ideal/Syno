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
    sampler = Sampler("[H,W]", "[H,W]", ["H:2", "W:2"], [
                      "s=3:2", "k=4:4"], net=net, depth=5, cuda=False, autoscheduler=CodeGenOptions.Li2018)
    mcts = MCTS(sampler, virtual_loss_constant=1)
    for idx in range(30):
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
    
    # nodes
    for k, v in mcts_recover._treenode_store.items():
        assert k in mcts._treenode_store, f"Node {k} not in {mcts._treenode_store}"
        assert v == mcts._treenode_store[k], f"Node {k} is {v}, should be {mcts._treenode_store[k]}"
    for k, v in mcts._treenode_store.items():
        if v.N == 0:
            continue
        assert k in mcts_recover._treenode_store, f"Node {k} not in {mcts_recover._treenode_store}"
        assert v == mcts_recover._treenode_store[k], f"Node {k} is {v}, should be {mcts_recover._treenode_store[k]}"
    
    # raves
    assert mcts.g_rave == mcts_recover.g_rave, f"Rave {mcts.g_rave} != {mcts_recover.g_rave}"
    
    # flags
    assert mcts._exploration_weight == mcts_recover._exploration_weight, f"Exploration weight {mcts._exploration_weight} != {mcts_recover._exploration_weight}"
    assert mcts.virtual_loss_constant == mcts_recover.virtual_loss_constant, f"Virtual loss constant {mcts.virtual_loss_constant} != {mcts_recover.virtual_loss_constant}"
    assert mcts.leaf_num == mcts_recover.leaf_num, f"Leaf num {mcts.leaf_num} != {mcts_recover.leaf_num}"
    assert mcts._b == mcts_recover._b, f"b {mcts._b} != {mcts_recover._b}"
    assert mcts._c_l == mcts_recover._c_l, f"c_l {mcts._c_l} != {mcts_recover._c_l}"
        
    os.remove("test_mcts.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mcts()
