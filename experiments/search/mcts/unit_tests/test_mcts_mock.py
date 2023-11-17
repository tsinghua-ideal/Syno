import json
import os, sys

from KAS import MockSampler
from kas_cpp_bindings import Next

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir
    )
)
from search.mcts.tree import MCTSTree
from search.mcts.node import TreePath


def test_remove():
    # If final nodes are all removed, then rollout return None (exhausted)
    vertices = ["root", {"name": "final", "is_final": True}]
    edges = [("root", [("Merge(1)", "final"), ("Shift(2)", "final")])]
    sampler = MockSampler(vertices, edges)

    mcts = MCTSTree(sampler, virtual_loss_constant=1)

    path_to_final = TreePath.decode_str("[Merge(1)]")
    mcts._increment_virtual_loss(path_to_final)
    mcts.remove(path_to_final, mcts.visit(path_to_final, on_tree=False))

    assert mcts.do_rollout() is None

    # If one final nodes is removed, then mcts should be able to find another path if any
    vertices = [
        "root",
        {"name": "final1", "is_final": True},
        {"name": "final2", "is_final": True},
    ]
    edges = [("root", [("Merge(1)", "final1"), ("Shift(2)", "final2")])]
    sampler = MockSampler(vertices, edges)

    mcts = MCTSTree(sampler, virtual_loss_constant=1)

    path_to_final = TreePath.decode_str("[Merge(1)]")
    mcts._increment_virtual_loss(path_to_final)
    mcts.remove(path_to_final, mcts.visit(path_to_final, on_tree=False))

    result = mcts.do_rollout()
    assert result[1][0][1]._node._node._name == "final2", result[1][0][
        1
    ]._node._node._name

    print("[PASSED] test_remove")


def test_virtual_loss() -> None:
    vertices = ["root", *[{"name": f"final{i+1}", "is_final": True} for i in range(6)]]
    edges = [
        (
            "root",
            [
                ("Merge(1)", "final1"),
                ("Merge(2)", "final2"),
                ("Merge(3)", "final3"),
                ("Unfold(3)", "final4"),
                ("Unfold(4)", "final5"),
                ("Unfold(5)", "final6"),
            ],
        )
    ]
    sampler = MockSampler(vertices, edges)

    mcts = MCTSTree(sampler, virtual_loss_constant=1, b=1)

    receipt, trial = mcts.do_rollout()
    mcts.back_propagate(receipt, 0.7, trial[0][0])
    receipt, trial = mcts.do_rollout()
    mcts.back_propagate(receipt, 0.7, trial[0][0])
    receipt, trial = mcts.do_rollout()
    mcts.back_propagate(receipt, 0.7, trial[0][0])

    trials = [mcts.do_rollout()[1] for _ in range(2)]

    assert len(set([trial[0][1]._node._node._name for trial in trials])) == 2, [
        trial[0][1]._node._node._name for trial in trials
    ]

    print("[PASSED] test_virtual_loss")


def test_exhausted():
    # If final nodes are all back proped, then rollout return None (exhausted)
    vertices = ["root", {"name": "final", "is_final": True}]
    edges = [("root", [("Merge(1)", "final"), ("Shift(2)", "final")])]
    sampler = MockSampler(vertices, edges)

    mcts = MCTSTree(sampler, virtual_loss_constant=1, b=1)

    receipt, trials = mcts.do_rollout()
    mcts.back_propagate(receipt, 0.9, trials[0][0])
    receipt, trials = mcts.do_rollout()
    mcts.back_propagate(receipt, 0.9, trials[0][0])
    receipt, trials = mcts.do_rollout()
    mcts.back_propagate(receipt, 0.9, trials[0][0])

    flag = mcts.do_rollout()
    assert flag is None, flag
    print("[PASSED] test_exhausted")


def test_reveal(length=10):
    vertices = [
        "root",
        *[{"name": f"final{i+1}", "is_final": True} for i in range(length)],
    ]
    edges = [("root", [(f"Finalize({i+1})", f"final{i+1}") for i in range(length)])]
    sampler = MockSampler(vertices, edges)
    mcts = MCTSTree(sampler, virtual_loss_constant=1)
    root = mcts.tree_root.get_children(auto_initialize=True)[0][1]
    root.get_children(auto_initialize=True)
    assert root.children_count(on_tree=True) == 0, root.children_count(True)
    assert (
        root.children_count(include_uninitialize=True) == length
    ), root.children_count(False)

    for iter in range(length):
        assert root.reveal_new_children()
        assert (
            root.children_count(on_tree=True) == iter + 1
        ), f"children_count={root.children_count(on_tree=True)} while iter={iter}"
    print("[PASSED] test_reveal")


def test_mcts():
    vertices = [
        "root",
        "a",
        "b",
        "c",
        *[{"name": f"final{i+1}", "is_final": True} for i in range(3)],
    ]
    edges = [
        ("root", [("Merge(1)", "a"), ("Shift(2)", "b")]),
        ("a", [("Shift(2)", "c")]),
        ("b", [("Merge(1)", "c")]),
        ("c", [(f"Finalize({i+3})", f"final{i+1}") for i in range(3)]),
    ]
    sampler = MockSampler(vertices, edges)

    mcts = MCTSTree(sampler, virtual_loss_constant=1)

    assert mcts.tree_root.children_count() == 2

    for _ in range(2):
        receipt, trials = mcts.do_rollout()
        assert (
            mcts.tree_root.children_count(on_tree=True) == 1
        ), mcts.tree_root.children_count(on_tree=True)
        mcts.back_propagate(receipt, 0.5, trials[0][0])

    assert len([v for _, v in mcts._treenode_store.items() if v.N > 0]) == 2

    mcts.garbage_collect()

    receipts = []
    trialss = []
    for _ in range(2):
        receipt, trials = mcts.do_rollout()
        receipts.append(receipt)
        trialss.append(trials)

    for tree_node in mcts._treenode_store.values():
        assert (
            not tree_node._isin_tree
        ) or tree_node._virtual_loss > 0, (
            f"Virtual loss count for {tree_node} is {tree_node._virtual_loss}"
        )

    json.dump(mcts.serialize(), open("test_mcts.json", "w"), indent=4)
    mcts_serialize = json.load(open("test_mcts.json", "r"))
    mcts_recover = MCTSTree.deserialize(mcts_serialize, sampler, keep_virtual_loss=True)

    mcts.back_propagate(receipts[0], 0.6, trialss[0][0][0])
    mcts.back_propagate(receipts[1], 0.6, trialss[1][0][0])

    for tree_node in mcts._treenode_store.values():
        assert (
            tree_node._virtual_loss == 0
        ), f"Virtual loss count for {tree_node} is {tree_node._virtual_loss}"
    assert len([v for _, v in mcts._treenode_store.items() if v.N > 0]) == 2

    # Test serialize
    json.dump(mcts.serialize(), open("test_mcts.json", "w"), indent=4)
    mcts_serialize = json.load(open("test_mcts.json", "r"))
    mcts_recover = MCTSTree.deserialize(mcts_serialize, sampler)
    for tree_node, v in mcts_recover._treenode_store.items():
        assert (
            tree_node in mcts._treenode_store
        ), f"Node {tree_node} not in {mcts._treenode_store}"
        assert v.eq_state(
            mcts._treenode_store[tree_node]
        ), f"Node {tree_node} is {v.state_dict}, should be {mcts._treenode_store[tree_node].state_dict}"
    for tree_node, v in mcts._treenode_store.items():
        if v.empty():
            continue
        assert (
            tree_node in mcts_recover._treenode_store
        ), f"Node {tree_node} not in {mcts_recover._treenode_store}"
        assert v.eq_state(
            mcts_recover._treenode_store[tree_node]
        ), f"Node {tree_node} is {v}, should be {mcts_recover._treenode_store[tree_node]}"

    # raves
    for tree_node, v in mcts.g_rave.items():
        assert (
            v == mcts_recover.g_rave[tree_node]
        ), f"{tree_node}: {v} != {mcts_recover.g_rave[tree_node]}"
    for tree_node, v in mcts_recover.g_rave.items():
        assert (
            v == mcts.g_rave[tree_node]
        ), f"{tree_node}: {v} != {mcts_recover.g_rave[tree_node]}"
    assert (
        mcts.g_rave == mcts_recover.g_rave
    ), f"Rave {mcts.g_rave} != {mcts_recover.g_rave}"

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


def test_rave():
    vertices = [
        "root",
        "s_1",
        "s_2",
        "s_3",
        {"name": "final_12", "is_final": True},
        {"name": "final_23", "is_final": True},
    ]
    edges = [
        ("root", [("Shift(1)", "s_1"), ("Shift(2)", "s_2"), ("Shift(3)", "s_3")]),
        ("s_1", [("Shift(2)", "final_12")]),
        ("s_3", [("Shift(2)", "final_23")]),
        ("s_2", [("Shift(1)", "final_12"), ("Shift(3)", "final_23")]),
    ]
    sampler = MockSampler(vertices, edges)

    mcts = MCTSTree(sampler, c_l=1e-4, b=0.9)

    # Extract all nodes
    root_node = mcts.tree_root
    share_level1 = root_node.get_children(auto_initialize=True, on_tree=False)[0][1]
    share_level1.reveal_new_children()
    share_level1.reveal_new_children()
    assert share_level1.children_count() == 3
    s_1 = share_level1.get_children(auto_initialize=True, on_tree=False)[0][1]
    s_2 = share_level1.get_children(auto_initialize=True, on_tree=False)[1][1]
    s_3 = share_level1.get_children(auto_initialize=True, on_tree=False)[2][1]
    assert s_1._node._node._name == "s_1"
    assert s_2._node._node._name == "s_2"
    assert s_3._node._node._name == "s_3"

    s_2.reveal_new_children()
    s_2_share = s_2.get_children(auto_initialize=True, on_tree=False)[0][1]
    s_2_share.reveal_new_children()
    s_2_share.reveal_new_children()
    assert s_2_share.children_count(include_uninitialize=True) == 2, (
        s_2_share,
        s_2_share.children_count(),
    )
    final_1 = s_2_share.get_children(auto_initialize=True, on_tree=False)[0][1]
    final_2 = s_2_share.get_children(auto_initialize=True, on_tree=False)[1][1]
    assert final_1._node._node._name == "final_12"
    assert final_2._node._node._name == "final_23"

    # Stage 1
    path_to_final1 = TreePath.decode_str("[Shift(2), Shift(1)]")
    assert mcts.visit(path_to_final1, on_tree=False, put_in_tree=True) == final_1
    mcts._increment_virtual_loss(path_to_final1)
    mcts.back_propagate(path_to_final1, 0.5, path_to_final1)

    # g_rave
    assert mcts.g_rave[Next(Next.Shift, 1)].mean == 0.5, mcts.g_rave
    assert mcts.g_rave[Next(Next.Shift, 2)].mean == 0.5, mcts.g_rave
    assert mcts.g_rave[Next(Next.Shift, 3)].mean == 0, mcts.g_rave

    # l_rave
    assert share_level1.l_rave[Next(Next.Shift, 1)].mean == 0.5, share_level1.l_rave
    assert share_level1.l_rave[Next(Next.Shift, 2)].mean == 0.5, share_level1.l_rave
    assert share_level1.l_rave[Next(Next.Shift, 3)].mean == 0, share_level1.l_rave
    assert s_1.l_rave[Next.Shift].mean == 0.5, s_1.l_rave
    assert s_2_share.l_rave[Next(Next.Shift, 1)].mean == 0.5, s_2_share.l_rave

    # Stage 2
    path_to_final2 = TreePath.decode_str("[Shift(2), Shift(3)]")
    mcts._increment_virtual_loss(path_to_final2)
    mcts.back_propagate(path_to_final2, 0.7, path_to_final2)

    # g_rave
    assert mcts.g_rave[Next(Next.Shift, 1)].mean == 0.5, mcts.g_rave
    assert mcts.g_rave[Next(Next.Shift, 2)].mean == 0.6, mcts.g_rave
    assert mcts.g_rave[Next(Next.Shift, 3)].mean == 0.7, mcts.g_rave

    # l_rave
    assert share_level1.l_rave[Next(Next.Shift, 1)].mean == 0.5, share_level1.l_rave
    assert share_level1.l_rave[Next(Next.Shift, 2)].mean == 0.6, share_level1.l_rave
    assert share_level1.l_rave[Next(Next.Shift, 3)].mean == 0.7, share_level1.l_rave
    assert s_3.l_rave[Next.Shift].mean == 0.7, s_1.l_rave
    assert s_2_share.l_rave[Next(Next.Shift, 3)].mean == 0.7, s_2_share.l_rave

    print("[PASSED] test_rave")


def test_converge(num_iter=1000, leaf_num=3, eps=0.03):
    """
    Here we construct a 3 level fully connected DAG. The result at root should converge to the best reward given infinite time.
    Convergence is no longer guaranteed now.
    """
    return
    vertices = [
        "root",
        "l1-1",
        "l1-2",
        "l1-3",
        "l2-1",
        "l2-2",
        "l2-3",
        "l3-1",
        "l3-2",
        "l3-3",
        {"name": "f1", "is_final": True, "reward": 0.2},
        {"name": "f2", "is_final": True, "reward": 0.9},
        {"name": "f3", "is_final": True, "reward": 0.6},
    ]
    edges = [
        *[("root", [(f"Merge(1{i})", f"l1-{i}") for i in [1, 2, 3]])],
        *[
            (f"l1-{j}", [(f"Unfold(1{j}2{k})", f"l2-{k}") for k in [1, 2, 3]])
            for j in [1, 2, 3]
        ],
        *[
            (f"l2-{j}", [(f"Split(2{j}3{k})", f"l3-{k}") for k in [1, 2, 3]])
            for j in [1, 2, 3]
        ],
        *[
            (f"l3-{j}", [(f"Shift(3{j}4{k})", f"f{k}") for k in [1, 2, 3]])
            for j in [1, 2, 3]
        ],
    ]
    sampler = MockSampler(vertices, edges)
    mcts = MCTSTree(sampler, virtual_loss_constant=1, leaf_num=leaf_num)

    from tqdm import trange

    for _ in trange(num_iter):
        receipt, trials = mcts.do_rollout(check_exhaustion=False)
        for path, node in trials:
            mcts.back_propagate(receipt, node._node.mock_get("reward"), path)

    root = mcts.tree_root
    assert (
        root.N <= num_iter * leaf_num + 1
    ), f"Root node has {root.N} visits, should be at most {num_iter * leaf_num + 1}"
    assert (
        abs(root.mean - 0.9) <= eps
    ), f"Q/N of root is {root.mean}, which has absolute error {abs(root.mean - 0.9)} > {eps}"

    print("[PASSED] test_convergence")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    test_remove()
    test_virtual_loss()
    test_exhausted()
    test_reveal()
    test_mcts()
    test_rave()
    test_converge()
