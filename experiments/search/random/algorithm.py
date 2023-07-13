from KAS.Node import VisitedNode, Path


class RandomAlgorithm:
    max_iterations = 2000
    max_final_iterations = 100000

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
            node = None
            for i in range(self.max_final_iterations):
                node = self.sampler.random_node_with_prefix(Path([]))
                if node.is_final():
                    break
            if node is None or not node.is_final():
                return 'end'

            assert isinstance(node, VisitedNode)
            path = node.path.serialize()
            if path not in self.sampled_paths:
                self.sampled_paths.add(path)
                return [path]

            # To many tries, the search space may be exhausted
            if n_iterations > self.max_iterations:
                return 'end'
