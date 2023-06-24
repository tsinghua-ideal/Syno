import logging
import json
import os
from tqdm import trange

from KAS import MCTS, MockSampler

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
    mcts.back_propagate(receipt, 0.5)
    mcts.back_propagate(receipt, 0.5)
    
    receipt, trials = mcts.do_rollout(sampler.root()) # root->Merge
    _, path = receipt
    node = trials
    print(f"Sampled {node} for {path}")
    mcts.back_propagate(receipt, 0.5)
    mcts.back_propagate(receipt, 0.5)
    
    mcts.do_rollout(sampler.root())

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
        node = trials
        print(f"Iteration {idx}. Sampled {node} for {path}")
        mcts.back_propagate(receipt, 0.5)
        
    print("Tree after first two iterations", [v for _, v in mcts._treenode_store.items() if v.N > 0])
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
    
    mcts.back_propagate(receipts[0], 0.6)
    mcts.back_propagate(receipts[1], 0.6)
    
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
        for _, node in trials:
            mcts.back_propagate(receipt, node._node.mock_get('reward'))
        
        root = mcts._treenode_store[sampler.visit([]).to_node()]
    
    root = mcts._treenode_store[sampler.visit([]).to_node()]
    assert root.N == num_iter * leaf_num, f"Root node has {root.N} visits, should be {num_iter * leaf_num}"
    assert abs(root.mean - 0.9) <= eps, f"Q/N of root is {root.mean}, which has absolute error {abs(root.mean - 0.9)} > {eps}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mcts()
    test_remove()
    test_final_select()
    test_converge()