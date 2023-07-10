import logging
import json
import os
from tqdm import trange

from KAS import MCTS, MockSampler
from kas_cpp_bindings import Next

def test_remove():
    
    # If final nodes are all removed, then rollout return None (exhausted)
    vertices = ['root', {'name': 'final', 'is_final': True}]
    edges = [
        ('root', [('Merge(1)', 'final'), ('Share(2)', 'final')])
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, virtual_loss_constant=1)
    
    receipt, trials = mcts.do_rollout(sampler.root()) # root->Merge
    _, path = receipt
    node = trials
    print(f"Sampled {node} for {path}")
    mcts.remove(receipt, trials[0][1])
    
    assert mcts.do_rollout(sampler.root()) is None
    
    # If one final nodes is removed, then mcts should be able to find another path if any
    vertices = ['root', {'name': 'final1', 'is_final': True}, {'name': 'final2', 'is_final': True}]
    edges = [
        ('root', [('Merge(1)', 'final1'), ('Share(2)', 'final2')])
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, virtual_loss_constant=1)
    
    receipt, trials = mcts.do_rollout(sampler.root()) # root->Merge
    _, path = receipt
    print(f"Sampled {trials} for {path}")
    mcts.remove(receipt, trials[0][1])
    
    assert mcts.do_rollout(sampler.root())[1][0][1]._node._node._name == 'final1' if trials[0][1]._node._node._name == 'final2' else 'final2'
    
    print("[PASSED] test_remove")
    
def test_virtual_loss() -> None:
    vertices = ['root', *[{'name': f'final{i+1}', 'is_final': True} for i in range(6)]]
    edges = [
        ('root', [('Merge(1)', 'final1'), ('Merge(2)', 'final2'), ('Merge(3)', 'final3'), ('Unfold(3)', 'final4'), ('Unfold(4)', 'final5'), ('Unfold(5)', 'final6')])
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, virtual_loss_constant=1, b=1)
    
    receipt, trial = mcts.do_rollout(sampler.root())
    mcts.back_propagate(receipt, 0.7, trial[0][0])
    receipt, trial = mcts.do_rollout(sampler.root())
    mcts.back_propagate(receipt, 0.7, trial[0][0])
    receipt, trial = mcts.do_rollout(sampler.root())
    mcts.back_propagate(receipt, 0.7, trial[0][0])
    
    trials = [mcts.do_rollout(sampler.root())[1] for _ in range(2)]
    
    assert len(set([trial[0][1]._node._node._name for trial in trials])) == 2, [trial[0][1]._node._node._name for trial in trials]
    
    print("[PASSED] test_virtual_loss")
    
def test_exhausted():
    
    # If final nodes are all back proped, then rollout return None (exhausted)
    vertices = ['root', {'name': 'final', 'is_final': True}]
    edges = [
        ('root', [('Merge(1)', 'final'), ('Share(2)', 'final')])
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, virtual_loss_constant=1, b=1)
    
    receipt, trials = mcts.do_rollout(sampler.root()) # root->Merge
    _, path = receipt
    node = trials
    print(f"Sampled {node} for {path}")
    mcts.back_propagate(receipt, .9, trials[0][0])
    
    receipt, trials = mcts.do_rollout(sampler.root()) # root->Share
    _, path = receipt
    node = trials
    print(f"Sampled {node} for {path}")
    mcts.back_propagate(receipt, .9, trials[0][0])
    assert mcts.tree_root.get_child(Next.Merge, mcts._treenode_store)[0].N == 1
    
    assert mcts.do_rollout(sampler.root()) is None
    print("[PASSED] test_exhausted")

def test_mcts():
    vertices = ['root', 'a', 'b', 'c', *[{'name': f'final{i+1}', 'is_final': True} for i in range(3)]]
    edges = [
        ('root', [('Merge(1)', 'a'), ('Share(2)', 'b')]),
        ('a', [('Share(2)', 'c')]),
        ('b', [('Merge(1)', 'c')]),
        ('c', [(f'Finalize({i+3})', f'final{i+1}') for i in range(3)])
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, virtual_loss_constant=1)
    
    assert mcts.tree_root.children_count(mcts._treenode_store) == 2
    receipt, trials = mcts.do_rollout(sampler.root())
    mcts.back_propagate(receipt, 0.5, trials[0][0])
    
    for idx in range(2):
        receipt, trials = mcts.do_rollout(sampler.root())
        assert mcts.tree_root.children_count(mcts._treenode_store, on_tree=True) == 1, mcts.tree_root.children_count(mcts._treenode_store, on_tree=True)
        _, path = receipt
        print(f"Iteration {idx}. Sampled {trials} for {path}")
        mcts.back_propagate(receipt, 0.5, trials[0][0])
        
    print("Tree after first two iterations", [(v, v.N) for _, v in mcts._treenode_store.items()])
    assert len([v for _, v in mcts._treenode_store.items() if v.N > 0]) == 2
    
    print(f"Garbage collection: size={len(mcts._treenode_store.keys())}->", end="")
    mcts.garbage_collect()
    print(len(mcts._treenode_store.keys()))
    
    receipts = []
    trialss = []
    for idx in range(2):
        receipt, trials = mcts.do_rollout(sampler.root())
        _, path = receipt
        print(f"Iteration {idx}. Sampled {trials} for {path}")
        receipts.append(receipt)
        trialss.append(trials)
    
    print("Virtual losses", mcts.virtual_loss_count.items())
    for k, v in mcts.virtual_loss_count.items():
        assert v > 0, f"Virtual loss count for {k} is {v}"
    
    mcts.back_propagate(receipts[0], 0.6, trialss[0][0][0])
    mcts.back_propagate(receipts[1], 0.6, trialss[1][0][0])
    
    for k, v in mcts.virtual_loss_count.items():
        assert v == 0, f"Virtual loss count for {k} is {v}"
    assert len([v for _, v in mcts._treenode_store.items() if v.N > 0]) == 2
    
    # Test serialize
    json.dump(mcts.serialize(), open("test_mcts.json", "w"), indent=4)
    mcts_serialize = json.load(open("test_mcts.json", "r"))
    mcts_recover = MCTS.deserialize(mcts_serialize, sampler)
    print("Original tree", mcts._treenode_store)
    print("Recovered tree", mcts_recover._treenode_store.items())
    for k, v in mcts_recover._treenode_store.items():
        assert k in mcts._treenode_store, f"Node {k} not in {mcts._treenode_store}"
        assert v == mcts._treenode_store[k], f"Node {k} is {v}, should be {mcts._treenode_store[k]}"
    for k, v in mcts._treenode_store.items():
        if v.empty():
            continue
        assert k in mcts_recover._treenode_store, f"Node {k} not in {mcts_recover._treenode_store}"
        assert v == mcts_recover._treenode_store[k], f"Node {k} is {v}, should be {mcts_recover._treenode_store[k]}"
    
    os.remove("test_mcts.json")
    
    print("[PASSED] test_mcts")

def test_grave():
    vertices = ['root', 's_1', 's_2', 's_3', {'name': 'final_12', 'is_final': True}, {'name': 'final_23', 'is_final': True}]
    edges = [
        ('root', [('Share(1)', 's_1'), ('Share(2)', 's_2'), ('Share(3)', 's_3')]),
        ('s_1', [('Share(2)', 'final_12')]),
        ('s_3', [('Share(2)', 'final_23')]),
        ('s_2', [('Share(1)', 'final_12'), ('Share(3)', 'final_23')]),
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, c_l=1e4, b=0.9)
    
    # s_1 is first expanded
    # CHECK: only one child
    receipt, trials = mcts.do_rollout(sampler.root())
    _, path = receipt
    print(f"Sampled {trials} for {path}")
    mcts.back_propagate(receipt, 0.1, trials[0][0])
    
    root_node = mcts._treenode_store[mcts._root]
    root_visible_children = root_node.get_children(mcts._treenode_store, auto_initialize=False, on_tree=True)
    assert len(root_visible_children) == 1
    
    # update once
    receipt, trials = mcts.do_rollout(sampler.root())
    _, path = receipt
    print(f"Sampled {trials} for {path}")
    mcts.back_propagate(receipt, 0.2, trials[0][0])
    
    # CHECK: no new children added. 
    root_visible_children = root_node.get_children(mcts._treenode_store, auto_initialize=False, on_tree=True)
    assert len(root_visible_children) == 1
    
    root_visible_grandchildren = root_visible_children[0][1].get_children(mcts._treenode_store, auto_initialize=False, on_tree=True)
    assert len(root_visible_grandchildren) == 1
    root_visible_grandchild = root_visible_grandchildren[0][1]
    assert root_visible_grandchild._node._node._name == 's_1'
    
    # update once
    receipt, trials = mcts.do_rollout(sampler.root())
    _, path = receipt
    print(f"Sampled {trials} for {path}")
    mcts.back_propagate(receipt, 0.5, trials[0][0])
    receipt, trials = mcts.do_rollout(sampler.root())
    _, path = receipt
    print(f"Sampled {trials} for {path}")
    mcts.back_propagate(receipt, 0.5, trials[0][0])
    
    # CHECK: two children from root
    root_visible_grandchildren = root_visible_children[0][1].get_children(mcts._treenode_store, auto_initialize=False, on_tree=True)
    assert len(root_visible_grandchildren) == 2, root_visible_grandchildren
    root_visible_grandchild = root_visible_grandchildren[1][1]
    assert root_visible_grandchild._node._node._name == 's_2'
    
    print("[PASSED] test_grave")

def test_lrave():
    vertices = ['root', 's_1', 's_2', 's_3', {'name': 'final_12', 'is_final': True}, {'name': 'final_23', 'is_final': True}]
    edges = [
        ('root', [('Share(1)', 's_1'), ('Share(2)', 's_2'), ('Share(3)', 's_3')]),
        ('s_1', [('Share(2)', 'final_12')]),
        ('s_3', [('Share(2)', 'final_23')]),
        ('s_2', [('Share(1)', 'final_12'), ('Share(3)', 'final_23')]),
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, c_l=1e-4, b=0.9)
    
    # s_1 is first expanded
    # CHECK: only one child
    receipt, trials = mcts.do_rollout(sampler.root())
    _, path = receipt
    print(f"Sampled {trials} for {path}")
    mcts.back_propagate(receipt, 0.1, trials[0][0])
    root_node = mcts._treenode_store[mcts._root]
    root_visible_children = root_node.get_children(mcts._treenode_store, auto_initialize=False, on_tree=True)
    assert len(root_visible_children) == 1
    
    # update once
    receipt, trials = mcts.do_rollout(sampler.root())
    _, path = receipt
    print(f"Sampled {trials} for {path}")
    mcts.back_propagate(receipt, 0., trials[0][0])
    
    # CHECK: no new children added. 
    root_visible_children = root_node.get_children(mcts._treenode_store, auto_initialize=False, on_tree=True)
    assert len(root_visible_children) == 1
    
    root_visible_grandchildren = root_visible_children[0][1].get_children(mcts._treenode_store, auto_initialize=False, on_tree=True)
    assert len(root_visible_grandchildren) == 1
    root_visible_grandchild = root_visible_grandchildren[0][1]
    assert root_visible_grandchild._node._node._name == 's_1'
    
    # update once
    receipt, trials = mcts.do_rollout(sampler.root())
    _, path = receipt
    print(f"Sampled {trials} for {path}")
    mcts.back_propagate(receipt, 0.5, trials[0][0])
    receipt, trials = mcts.do_rollout(sampler.root())
    _, path = receipt
    print(f"Sampled {trials} for {path}")
    mcts.back_propagate(receipt, 0.5, trials[0][0])
    
    # CHECK: two children from root
    root_visible_grandchildren = root_visible_children[0][1].get_children(mcts._treenode_store, auto_initialize=False, on_tree=True)
    assert len(root_visible_grandchildren) == 2, root_visible_grandchildren
    root_visible_grandchild = root_visible_grandchildren[1][1]
    assert root_visible_grandchild._node._node._name == 's_2'
    
    print("[PASSED] test_lrave")

def test_converge(num_iter=1000, leaf_num=3, eps=0.03):
    """
    Here we construct a 3 level fully connected DAG. The result at root should converge to the best reward given infinite time. 
    """
    vertices = [
        'root', 
        'l1-1', 'l1-2', 'l1-3', 
        'l2-1', 'l2-2', 'l2-3', 
        'l3-1', 'l3-2', 'l3-3', 
        {'name': 'f1', 'is_final': True, 'reward': 0.2}, 
        {'name': 'f2', 'is_final': True, 'reward': 0.9}, 
        {'name': 'f3', 'is_final': True, 'reward': 0.6}
    ]
    edges = [
        *[('root', [(f'Merge(1{i})', f'l1-{i}') for i in [1, 2, 3]])],
        *[(f'l1-{j}', [(f'Unfold(1{j}2{k})', f'l2-{k}') for k in [1, 2, 3]]) for j in [1, 2, 3]],
        *[(f'l2-{j}', [(f'Split(2{j}3{k})', f'l3-{k}') for k in [1, 2, 3]]) for j in [1, 2, 3]],
        *[(f'l3-{j}', [(f'Share(3{j}4{k})', f'f{k}') for k in [1, 2, 3]]) for j in [1, 2, 3]]
    ]
    sampler = MockSampler(vertices, edges)
    mcts = MCTS(sampler, virtual_loss_constant=1, leaf_num=leaf_num)
    
    for _ in trange(num_iter):
        receipt, trials = mcts.do_rollout(sampler.root(), check_exhaustion=False)
        for path, node in trials:
            mcts.back_propagate(receipt, node._node.mock_get('reward'), path)
    
    root = mcts.tree_root
    assert root.N == num_iter * leaf_num, f"Root node has {root.N} visits, should be {num_iter * leaf_num}"
    assert abs(root.mean - 0.9) <= eps, f"Q/N of root is {root.mean}, which has absolute error {abs(root.mean - 0.9)} > {eps}"
    
    print("[PASSED] test_convergence")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_remove()
    test_virtual_loss()
    test_exhausted()
    test_mcts()
    test_grave()
    test_lrave()
    test_converge()