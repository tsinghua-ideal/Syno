import logging
import json
import traceback

from KAS import MCTS, MockSampler

def line_sampler():
    "A diamond"
    vertices = ['root', {'name': 'final', 'is_final': True}]

    edges = [
        ('root', [('Merge(1)', 'final'), ('Share(2)', 'final')])
    ]
    
    sampler = MockSampler(vertices, edges)
    return sampler

def diamond_sampler():
    "A diamond"
    vertices = ['root', 'a', 'b', {'name': 'final', 'is_final': True}]

    edges = [
        ('root', [('Merge(1)', 'a'), ('Share(2)', 'b')]),
        ('a', [('Split(3)', 'final')]),
        ('b', [('Unfold(4)', 'final')]),
    ]
    
    sampler = MockSampler(vertices, edges)
    return sampler

def test_remove():
    sampler = line_sampler()
    mcts = MCTS(sampler, virtual_loss_constant=1)
    
    receipt, trials = mcts.do_rollout(sampler.root()) # root->Merge
    _, path = receipt
    node = trials
    print(f"Sampled {node} for {path}:")
    mcts.remove(receipt, trials[0])
    
    assert mcts.do_rollout(sampler.root()) is None # 

def test_mcts():
    sampler = diamond_sampler()
    mcts = MCTS(sampler, virtual_loss_constant=1)
    
    assert mcts._treenode_store[sampler.root().to_node()].children_count(mcts._treenode_store) == 2, mcts._treenode_store
    
    for idx in range(2):
        receipt, trials = mcts.do_rollout(sampler.root())
        _, path = receipt
        node = trials
        print(f"Iteration {idx}. Sampled {node} for {path}")
        mcts.back_propagate(receipt, 0.5)
        
    print("Tree after first two iterations", [v for _, v in mcts._treenode_store.items() if v.N > 0])
    assert len(mcts._treenode_store.keys()) == 4
    assert len([v for _, v in mcts._treenode_store.items() if v.N > 0]) == 1
    
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
    assert len([v for _, v in mcts._treenode_store.items() if v.N > 0]) == 3
    
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mcts()
    test_remove()
