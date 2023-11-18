import json
import os
import sys
import matplotlib.pyplot as plt
import logging
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from base import log, models, parser, parser
from search import MCTSTree, MCTSAlgorithm


if __name__ == "__main__":
    log.setup(logging.INFO)

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

    def rel_error(total_value, total_N, edge_state_value, edge_state_N):
        assert total_N >= edge_state_N
        estimated_value = (total_value * total_N - edge_state_value * edge_state_N) / (
            total_N - edge_state_N
        )
        return estimated_value - edge_state_value  # / edge_state_value

    avg_mean, avg_std = [], []
    for node in mcts._treenode_store.values():
        if node == mcts.tree_root:
            continue
        for nxt, child, edge_state in node.get_children(on_tree=True):
            l_rave = node.l_rave[nxt]
            if (
                l_rave.N > edge_state.N >= 10
                and l_rave.mean != 0
                and edge_state.mean != 0
            ):
                avg_mean.append(edge_state.mean)
                avg_std.append(edge_state.std)

    print(
        f"avg_mean={sum(avg_mean) / len(avg_mean)}, avg_std={sum(avg_std) / len(avg_std)}"
    )

    # L-rave
    avg_mean_diff, avg_std_diff = [], []
    for node in mcts._treenode_store.values():
        if node == mcts.tree_root:
            continue
        for nxt, child, edge_state in node.get_children(on_tree=True):
            l_rave = node.l_rave[nxt]
            if (
                l_rave.N > edge_state.N >= 10
                and l_rave.mean != 0
                and edge_state.mean != 0
            ):
                avg_mean_diff.append(
                    rel_error(l_rave.mean, l_rave.N, edge_state.mean, edge_state.N)
                )
                avg_std_diff.append(
                    rel_error(l_rave.std, l_rave.N, edge_state.std, edge_state.N)
                )

    plt.hist(avg_mean_diff, bins=50, density=True)
    plt.title("Distribution of error of l-rave mean")
    plt.savefig("analysis/results/l-rave-mean-distribution.png")
    plt.clf()

    plt.hist(avg_std_diff, bins=50, density=True)
    plt.title("Distribution of error of l-rave mean")
    plt.savefig("analysis/results/l-rave-std-distribution.png")
    plt.clf()

    avg_mean_diff = np.sort(np.abs(np.array(avg_mean_diff)))
    print(
        f"L-rave mean 50%: {avg_mean_diff[int(len(avg_mean_diff) * 0.5)]}, 80%: {avg_mean_diff[int(len(avg_mean_diff) * 0.8)]}, 90%: {avg_mean_diff[int(len(avg_mean_diff) * 0.9)]}, 95%: {avg_mean_diff[int(len(avg_mean_diff) * 0.95)]}, 99%: {avg_mean_diff[int(len(avg_mean_diff) * 0.99)]}"
    )
    print(f"L-rave mean < 0.03: {np.sum(avg_mean_diff < 0.03) / len(avg_mean_diff)}")

    avg_std_diff = np.sort(np.abs(np.array(avg_std_diff)))
    print(
        f"L-rave std 50%: {avg_std_diff[int(len(avg_std_diff) * 0.5)]}, 80%: {avg_std_diff[int(len(avg_std_diff) * 0.8)]}, 90%: {avg_std_diff[int(len(avg_std_diff) * 0.9)]}, 95%: {avg_std_diff[int(len(avg_std_diff) * 0.95)]}, 99%: {avg_std_diff[int(len(avg_std_diff) * 0.99)]}"
    )
    print(f"L-rave std < 0.03: {np.sum(avg_std_diff < 0.03) / len(avg_std_diff)}")

    # G-rave
    avg_mean_diff, avg_std_diff = [], []
    for node in mcts._treenode_store.values():
        for nxt, child, edge_state in node.get_children(on_tree=True):
            g_rave = mcts.g_rave[nxt]
            if (
                g_rave.N > edge_state.N >= 10
                and edge_state.mean != 0
                and g_rave.mean != 0
            ):
                avg_mean_diff.append(
                    rel_error(g_rave.mean, g_rave.N, edge_state.mean, edge_state.N)
                )
                avg_std_diff.append(
                    rel_error(g_rave.std, g_rave.N, edge_state.std, edge_state.N)
                )

    plt.figure()
    plt.hist(avg_mean_diff, bins=50, density=True)
    plt.title("Distribution of relative error of g-rave mean")
    plt.savefig("analysis/results/g-rave-mean-distribution.png")
    plt.clf()

    plt.figure()
    plt.hist(avg_std_diff, bins=50, density=True)
    plt.title("Distribution of relative error of g-rave mean")
    plt.savefig("analysis/results/g-rave-std-distribution.png")
    plt.clf()
