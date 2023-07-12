from KAS.Node import VisitedNode, Path


class RandomAlgorithm:
    max_iterations = 20

    def __init__(self, sampler, args):
        self.sampler = sampler
        self.sampled_paths = set()

    def serialize(self):
        return {'sampled': list(self.sampled_paths)}

    def update(self, path, reward):
        pass

    def sample(self):
        n_iterations = 0
        while True:
            # Random a node
            n_iterations += 1
            while True:
                node = self.sampler.random_node_with_prefix(Path([]))
                if node.is_final():
                    break
            assert isinstance(node, VisitedNode)

            path = node.path.serialize()
            if path not in self.sampled_paths:
                self.sampled_paths.add(path)
                return [path]

            # To many tries, the search space may be exhausted
            if n_iterations > self.max_iterations:
                return 'end'
