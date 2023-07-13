import logging
from KAS.Sampler import Sampler
from KAS.Node import Path


class BeamAlgorithm:
    max_queue_size = 300
    max_estimates = 8
    max_final_iterations = 100000

    def __init__(self, sampler: Sampler, args):
        self.sampler = sampler

        # Life of a path:
        #   pending -> waiting -> heap
        self.pending = set()
        self.waiting = dict()
        self.heap = []

        # For pushing check
        self.visited = set()

        # For estimating
        self.ancestor = dict()

        # Push root
        path = self.sampler.root().path.serialize()
        self.visited.add(path)
        self.pending.add(path)
        
    def serialize(self):
        # TODO: implement
        return {}

    def update(self, path, reward):
        # Update the estimate
        assert path in self.ancestor
        ancestor_path = self.ancestor[path]
        assert ancestor_path in self.waiting
        max_reward, estimates = self.waiting[ancestor_path]
        assert path in estimates
        max_reward = max(max_reward, reward)
        estimates.remove(path)

        # Update the heap
        if len(estimates) == 0:
            del self.waiting[ancestor_path]
            self.heap.append((max_reward, ancestor_path))
            self.heap = sorted(self.heap)
            
        # Maintain the heap
        if len(self.heap) > self.max_queue_size:
            self.heap.pop(0)
        assert len(self.heap) <= self.max_queue_size

    def sample(self):
        # Expand the heap
        if len(self.pending) == 0 and len(self.heap) > 0:
            _, top_path = self.heap.pop()
            top_node = self.sampler.visit(Path.deserialize(top_path))
            children = [top_node.get_child(next) for next in top_node.get_children_handles()]
            children = filter(lambda x: x is not None, children)
            for child in children:
                path = child.path.serialize()
                if path not in self.visited:
                    self.visited.add(path)
                    self.pending.add(path)
        
        while len(self.pending) > 0:
            serialized_path = self.pending.pop()
            path = Path.deserialize(serialized_path)
            node = self.sampler.visit(path)
            if node.is_final():
                self.waiting[serialized_path] = (0, set([serialized_path]))
                return [serialized_path]
            
            # Sample
            estimates = []
            for i in range(self.max_estimates):
                new_node = None
                for j in range(self.max_final_iterations):
                    node = self.sampler.random_node_with_prefix(path)
                    if node.is_final():
                        new_node = node
                        break
                if new_node:
                    serialized_new_path = new_node.path.serialize()
                    estimates.append(serialized_new_path)
                    self.ancestor[serialized_new_path] = serialized_path
            
            # Return if not empty
            if len(estimates) > 0:
                self.waiting[serialized_path] = (0, set(estimates))
                return estimates
            
        return 'retry'
        