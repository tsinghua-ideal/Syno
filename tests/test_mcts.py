import logging

from KAS import Sampler, MCTS, Path



def test_mcts():
    sampler = Sampler("[H,W]", "[H,W]", ["H = 128: 1", "W: 3"], ["s_1=2: 2", "k_1=3", "4"], depth=5)
    mcts = MCTS(sampler)
    for idx in range(10):
        node = mcts.do_rollout(sampler.root())
        path = node.path
        print(f"Iteration {idx}. Sampled {path}:")
        print(sampler.path_to_strs(path))
        for i in range(len(path)):
            child = sampler.visit(Path(path.abs_path[:i]))
            print(f"Node {child} has children:", child.get_children_types())
        mcts.back_propagate(node, 1.0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_mcts()
