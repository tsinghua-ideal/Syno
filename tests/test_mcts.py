import logging

from KAS import Sampler, MCTS


def test_mcts():
    sampler = Sampler("[H,W]", "[H,W]", ["H = 128: 1", "W: 3"], ["s_1=2: 2", "k_1=3", "4"], depth=5)
    mcts = MCTS(sampler)
    for idx in range(10):
        path = mcts.do_rollout([])
        sampler._realize(path)
        print(f"Iteration {idx}. Sampled {path}:")
        print(sampler._path_str(path))
        for i in range(len(path)):
            node = path[:i]
            print(f"Node {node} has children:", sampler.children_types(node))
        mcts.back_propagate(path, 1.0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_mcts()
