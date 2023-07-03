import logging
import json
import os
from tqdm import trange

from KAS import MCTS, MockSampler
from kas_cpp_bindings import Next

def test_remove():
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
    print("[PASSED] test_remove")
    
def test_final_select():
    
    vertices = ['root', {'name': 'final', 'is_final': True}]
    edges = [
        ('root', [('Merge(1)', 'final')])
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, virtual_loss_constant=1, leaf_num=2)
    
    receipt, trials = mcts.do_rollout(sampler.root()) # root->Merge
    _, path = receipt
    node = trials
    print(f"Sampled {node} for {path}")
    mcts.back_propagate(receipt, 0.5, node[0][0])
    mcts.back_propagate(receipt, 0.5, node[0][0])
    
    receipt, trials = mcts.do_rollout(sampler.root()) # root->Merge
    _, path = receipt
    node = trials
    print(f"Sampled {node} for {path}")
    mcts.back_propagate(receipt, 0.5, node[0][0])
    mcts.back_propagate(receipt, 0.5, node[0][0])
    
    mcts.do_rollout(sampler.root())
    
    print("[PASSED] test_final_select")

def test_mcts():
    vertices = ['root', 'a', 'b', {'name': 'final', 'is_final': True}]
    edges = [
        ('root', [('Merge(1)', 'a'), ('Share(2)', 'b')]),
        ('a', [('Split(3)', 'final')]),
        ('b', [('Unfold(4)', 'final')]),
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, virtual_loss_constant=1)
    
    assert mcts._treenode_store[sampler.root().to_node()].children_count(mcts._treenode_store) == 2, mcts._treenode_store[sampler.root().to_node()].children_count(mcts._treenode_store)
    
    for idx in range(2):
        receipt, trials = mcts.do_rollout(sampler.root())
        _, path = receipt
        print(f"Iteration {idx}. Sampled {trials} for {path}")
        mcts.back_propagate(receipt, 0.5, trials[0][0])
        print("Tree after first two iterations", [(v, v.N) for _, v in mcts._treenode_store.items()])
        
    print("Tree after first two iterations", [(v, v.N) for _, v in mcts._treenode_store.items()])
    assert len(mcts._treenode_store.keys()) == 3
    assert len([v for _, v in mcts._treenode_store.items() if v.N > 0]) == 2
    
    print(f"Garbage collection: size={len(mcts._treenode_store.keys())}->", end="")
    mcts.garbage_collect()
    print(len(mcts._treenode_store.keys()))
    
    receipts = []
    trialss = []
    for idx in range(2, 4):
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
    mcts_serialize = mcts.serialize()
    json.dump(mcts_serialize, open("test_mcts.json", "w"), indent=4)
    mcts_recover = MCTS.deserialize(mcts_serialize, sampler)
    print("Original tree", mcts._treenode_store)
    print("Recovered tree", mcts_recover._treenode_store.items())
    for k, v in mcts_recover._treenode_store.items():
        assert k in mcts._treenode_store, f"Node {k} not in {mcts._treenode_store}"
        assert v == mcts._treenode_store[k], f"Node {k} is {v}, should be {mcts._treenode_store[k]}"
    for k, v in mcts._treenode_store.items():
        if v.N == 0:
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
        ('s_2', [('Share(1)', 'final_12'), ('Share(3)', 'final_23')]),
        ('s_3', [('Share(2)', 'final_23')]),
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, c_l=1e4, b=0.9)
    
    # s_1 is first expanded
    # CHECK: only one child
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
    
    print("[PASSED] test_grave")

def test_grave():
    vertices = ['root', 's_1', 's_2', 's_3', {'name': 'final_12', 'is_final': True}, {'name': 'final_23', 'is_final': True}]
    edges = [
        ('root', [('Share(1)', 's_1'), ('Share(2)', 's_2'), ('Share(3)', 's_3')]),
        ('s_1', [('Share(2)', 'final_12')]),
        ('s_2', [('Share(1)', 'final_12'), ('Share(3)', 'final_23')]),
        ('s_3', [('Share(2)', 'final_23')]),
    ]
    sampler = MockSampler(vertices, edges)
    
    mcts = MCTS(sampler, c_l=1e-4, b=0.9)
    
    # s_1 is first expanded
    # CHECK: only one child
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
    
    print("[PASSED] test_grave")

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
        receipt, trials = mcts.do_rollout(sampler.root())
        for path, node in trials:
            mcts.back_propagate(receipt, node._node.mock_get('reward'), path)
        
        root = mcts._treenode_store[sampler.visit([]).to_node()]
    
    root = mcts._treenode_store[sampler.visit([]).to_node()]
    assert root.N == num_iter * leaf_num, f"Root node has {root.N} visits, should be {num_iter * leaf_num}"
    assert abs(root.mean - 0.9) <= eps, f"Q/N of root is {root.mean}, which has absolute error {abs(root.mean - 0.9)} > {eps}"
    
    print("[PASSED] test_convergence")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_remove()
    test_final_select()
    test_mcts()
    test_grave()
    test_converge()