import logging
import torch
import torch.nn as nn
import json
import traceback

from KAS import CodeGenOptions, Sampler, MCTS, Path, Placeholder, MockSampler

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel = Placeholder({"H": 128, "W": 128})

    def forward(self, x: torch.Tensor):
        x = self.kernel(x)
        return x

def test_mcts():
    net = Model()
    sampler = Sampler("[H,W]", "[H,W]", ["H = 128: 1", "W = 128: 1"], [
                      "s_1=2: 2", "k=3: 2"], net=net, depth=5, cuda=False, autoscheduler=CodeGenOptions.Li2018)
    mcts = MCTS(sampler, virtual_loss_constant=1)
    in_tensor = torch.randn((128, 128))
    for idx in range(10):
        try:
            receipt, trials = mcts.do_rollout(sampler.root())
            _, path = receipt
            node = trials[0]
            print(f"Iteration {idx}. Sampled {node} for {path}:")
            print(path.path_to_strs(sampler))
            s = node.path.serialize()
            d = node.path.deserialize(s)
            print("Serialized Path", s)
            print("Deserialized Path", d)
            for i in range(len(node.path)):
                child = sampler.visit(Path(node.path.abs_path[:i]))
                print(f"Node {child} has children:",
                      child.get_children_types())
            kernel_packs, _ = sampler.realize(net, node, f"test_mcts_{idx}")
            sampler.replace(net, kernel_packs)
            print(f"Computing forward {idx}...")
            print(f"Result: {net(in_tensor)}")
            mcts.back_propagate(receipt, 1.0)
        except Exception as e:
            print(f"Caught error {e}")
            traceback.print_exc()
    
    for k, v in mcts.virtual_loss_count.items():
        assert v == 0, f"Virtual loss count for {k} is {v}"
    
    # Test serialize
    mcts_serialize = mcts.serialize()
    json.dump(mcts_serialize, open("test_mcts.json", "w"), indent=4)
    mcts_recover = MCTS.deserialize(mcts_serialize, sampler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_mcts()
