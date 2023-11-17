import logging
import torch
import torch.nn as nn
import json
import os, sys
from random import random

from KAS import CodeGenOptions, Sampler, Placeholder


sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir
    )
)
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
    mcts = MCTSTree(sampler, virtual_loss_constant=1)
    for idx in range(30):
        receipt, trials = mcts.do_rollout()
        path = receipt
        final_pack = trials[0]
        print(f"Iteration {idx}. Sampled {final_pack[0]} for {path}")
        if random() <= backprop_prob:
            mcts.back_propagate(receipt, random(), final_pack[0])
        else:
            mcts.remove(receipt, final_pack[1])

    for tree_node in mcts._treenode_store.values():
        assert (
            tree_node.is_dead_end() or tree_node._virtual_loss == 0
        ), f"Virtual loss count for {tree_node} is {tree_node._virtual_loss}"

    receipts = []
    for idx in range(30):
        receipt, trials = mcts.do_rollout()
        path = receipt
        final_pack = trials[0]
        assert final_pack[1].is_final()
        assert sampler.visit(final_pack[0]).to_node() == final_pack[1].to_node()
        print(f"Iteration {idx}. Sampled {final_pack[0]} for {path}")
        receipts.append((final_pack, receipt))

    result_buffer = {}

    for final_pack, receipt in receipts:
        path = final_pack[0]
        node = final_pack[1].to_node()
        if node in result_buffer:
            if result_buffer[node] != -1:
                logging.info(f"Backpropagating {receipt} with {path}")
                mcts.back_propagate(receipt, result_buffer[node], path)
            else:
                mcts._decrement_virtual_loss(receipt)
        elif random() <= backprop_prob:
            result_buffer[node] = random()
            logging.info(
                f"Backpropagating {receipt} with {path} and {result_buffer[node]}"
            )
            mcts.back_propagate(receipt, result_buffer[node], path)
        else:
            result_buffer[node] = -1
            logging.info(f"Removing {receipt} with {path}")
            mcts.remove(receipt, final_pack[1])

    for tree_node in mcts._treenode_store.values():
        assert (
            tree_node.is_dead_end() or tree_node._virtual_loss == 0
        ), f"Virtual loss count for {tree_node} is {tree_node._virtual_loss}"

    # # Test serialize
    # print("Testing serialization and deserialization. ")
    # json.dump(mcts.serialize(), open("test_mcts.json", "w"), indent=4)
    # mcts_serialize = json.load(open("test_mcts.json", "r"))
    # mcts_recover = MCTSTree.deserialize(mcts_serialize, sampler, keep_dead_state=True)

    # # nodes
    # for tree_node, v in mcts_recover._treenode_store.items():
    #     assert (
    #         tree_node in mcts._treenode_store
    #     ), f"Node {tree_node} not in mcts._treenode_store"
    #     assert v.eq_state(
    #         mcts._treenode_store[tree_node]
    #     ), f"Node {tree_node.get_possible_path()} is {v.l_rave}, should be {mcts._treenode_store[tree_node].l_rave}"

    # raves
    # for k, v in mcts.g_rave.items():
    #     assert k in mcts_recover.g_rave, f"{k} does not exists in mcts_recover.g_rave"
    #     assert v == mcts_recover.g_rave[k], f"{k}: {v} != {mcts_recover.g_rave[k]}"
    # for tree_node, v in mcts_recover.g_rave.items():
    #     assert tree_node in mcts.g_rave, f"{tree_node} does not exists in mcts.g_rave"
    #     assert (
    #         v == mcts.g_rave[tree_node]
    #     ), f"{tree_node}: {v} != {mcts_recover.g_rave[tree_node]}"

    # # flags
    # assert (
    #     mcts._exploration_weight == mcts_recover._exploration_weight
    # ), f"Exploration weight {mcts._exploration_weight} != {mcts_recover._exploration_weight}"
    # assert (
    #     mcts.virtual_loss_constant == mcts_recover.virtual_loss_constant
    # ), f"Virtual loss constant {mcts.virtual_loss_constant} != {mcts_recover.virtual_loss_constant}"
    # assert (
    #     mcts.leaf_num == mcts_recover.leaf_num
    # ), f"Leaf num {mcts.leaf_num} != {mcts_recover.leaf_num}"
    # assert mcts._b == mcts_recover._b, f"b {mcts._b} != {mcts_recover._b}"
    # assert mcts._c_l == mcts_recover._c_l, f"c_l {mcts._c_l} != {mcts_recover._c_l}"

    # os.remove("test_mcts.json")
    print("[PASSED] test_mcts")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    test_mcts()
