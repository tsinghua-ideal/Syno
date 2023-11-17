import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from base import log, models, parser, parser
from search import MCTSTree, MCTSAlgorithm


if __name__ == "__main__":
    log.setup()

    # Arguments
    args = parser.arg_parse()

    # Sampler
    model, sampler = models.get_model(args, return_sampler=True)

    # Explorer
    if args.kas_mcts_explorer_path:
        with open(args.kas_mcts_explorer_path, "r") as file:
            mcts = MCTSTree.deserialize(
                json.load(file), sampler, keep_virtual_loss=False, keep_dead_state=False
            )
    else:
        mcts = MCTSAlgorithm(sampler, args).mcts

    # L-rave
    avg_mean_diff, avg_std_diff = [], []
    for node in mcts._treenode_store.values():
        for nxt, child, edge_state in node.get_children(on_tree=True):
            l_rave = node.l_rave[nxt]
            if l_rave.N >= 10 and edge_state.N >= 10:
                avg_mean_diff.append((l_rave.mean - edge_state.mean) / edge_state.mean)
                avg_std_diff.append((l_rave.std - edge_state.std) / edge_state.std)

    plt.hist(avg_mean_diff, bins=20)
    plt.title("Distribution of relative error of l-rave mean")
    plt.savefig("l-rave-mean-distribution.png")
    plt.clf()

    plt.hist(avg_std_diff, bins=20)
    plt.title("Distribution of relative error of l-rave mean")
    plt.savefig("l-rave-std-distribution.png")
    plt.clf()

    # G-rave
    avg_mean_diff, avg_std_diff = [], []
    for node in mcts._treenode_store.values():
        for nxt, child, edge_state in node.get_children(on_tree=True):
            g_rave = mcts.g_rave[nxt]
            if g_rave.N >= 10 and edge_state.N >= 10:
                avg_mean_diff.append((g_rave.mean - edge_state.mean) / edge_state.mean)
                avg_std_diff.append((g_rave.std - edge_state.std) / edge_state.std)

    plt.figure()
    plt.hist(avg_mean_diff, bins=20)
    plt.title("Distribution of relative error of g-rave mean")

    plt.figure()
    plt.hist(avg_std_diff, bins=20)
    plt.title("Distribution of relative error of g-rave mean")
