import logging
import torch
import torch.nn as nn
import json
import os, sys
from random import random

from KAS import CodeGenOptions, Sampler, Placeholder

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from search.mcts.tree import MCTSTree


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel = Placeholder({"H": 128, "W": 128})

    def forward(self, x: torch.Tensor):
        x = self.kernel(x)
        return x


def test_mcts(backprop_prob=0.5):
    net = Model()
    sampler = Sampler(
        "[H,W]",
        "[H,W]",
        ["H:2", "W:2"],
        ["s=3:2", "k=4:4"],
        net=net,
        depth=5,
        cuda=False,
        num_worker_threads=4,
        autoscheduler=CodeGenOptions.Li2018,
    )
    mcts = MCTSTree(sampler, virtual_loss_constant=1, simulate_retry_period=1e6)
    for idx in range(30):
        receipt, trials = mcts.do_rollout()
        path = receipt
        node = trials[0]
        print(f"Iteration {idx}. Sampled {node[0]} for {path}")
        if random() <= backprop_prob:
            mcts.back_propagate(receipt, random(), node[0])
        else:
            mcts.remove(receipt, node[1])

    for k, v in mcts.virtual_loss_count.items():
        assert k.is_dead_end() or v == 0, f"Virtual loss count for {k} is {v}"

    receipts = []
    for idx in range(30):
        receipt, trials = mcts.do_rollout()
        path = receipt
        node = trials[0]
        print(f"Iteration {idx}. Sampled {node[0]} for {path}")
        receipts.append((node, receipt))

    json.dump(mcts.serialize(), open("test_mcts.json", "w"), indent=4)
    mcts_serialize = json.load(open("test_mcts.json", "r"))
    mcts_recover = MCTSTree.deserialize(mcts_serialize, sampler, keep_virtual_loss=True)
    for k, v in mcts_recover.virtual_loss_count.items():
        assert v == mcts.virtual_loss_count[k], f"{v} != {mcts.virtual_loss_count[k]}"
    for k, v in mcts.virtual_loss_count.items():
        assert (
            k.is_final() or k.is_dead_end() or v == mcts_recover.virtual_loss_count[k]
        ), f"{k}: {v} != {mcts_recover.virtual_loss_count[k]}"

    for node, receipt in receipts:
        if random() <= backprop_prob:
            mcts.back_propagate(receipt, random(), node[0])
        else:
            mcts.remove(receipt, node[1])

    for k, v in mcts.virtual_loss_count.items():
        assert (
            k.is_final() or k.is_dead_end() or v == 0
        ), f"Virtual loss count for {k} is {v}"

    # Test serialize
    print("Testing serialization and deserialization. ")
    json.dump(mcts.serialize(), open("test_mcts.json", "w"), indent=4)
    mcts_serialize = json.load(open("test_mcts.json", "r"))
    mcts_recover = MCTSTree.deserialize(mcts_serialize, sampler)

    # nodes
    for k, v in mcts_recover._treenode_store.items():
        assert k in mcts._treenode_store, f"Node {k} not in mcts._treenode_store"
        assert v.eq_state(
            mcts._treenode_store[k]
        ), f"Node {k.get_possible_path()} is {v.l_rave}, should be {mcts._treenode_store[k].l_rave}"

    # raves
    # for k, v in mcts.g_rave.items():
    #     assert k in mcts_recover.g_rave, f"{k} does not exists in mcts_recover.g_rave"
    #     assert v == mcts_recover.g_rave[k], f"{k}: {v} != {mcts_recover.g_rave[k]}"
    for k, v in mcts_recover.g_rave.items():
        assert k in mcts.g_rave, f"{k} does not exists in mcts.g_rave"
        assert v == mcts.g_rave[k], f"{k}: {v} != {mcts_recover.g_rave[k]}"

    # flags
    assert (
        mcts._exploration_weight == mcts_recover._exploration_weight
    ), f"Exploration weight {mcts._exploration_weight} != {mcts_recover._exploration_weight}"
    assert (
        mcts.virtual_loss_constant == mcts_recover.virtual_loss_constant
    ), f"Virtual loss constant {mcts.virtual_loss_constant} != {mcts_recover.virtual_loss_constant}"
    assert (
        mcts.leaf_num == mcts_recover.leaf_num
    ), f"Leaf num {mcts.leaf_num} != {mcts_recover.leaf_num}"
    assert mcts._b == mcts_recover._b, f"b {mcts._b} != {mcts_recover._b}"
    assert mcts._c_l == mcts_recover._c_l, f"c_l {mcts._c_l} != {mcts_recover._c_l}"

    os.remove("test_mcts.json")
    print("[PASSED] test_mcts")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    test_mcts()
