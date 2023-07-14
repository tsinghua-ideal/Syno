import logging
from KAS.Sampler import Sampler
from KAS.Node import Path


class BeamAlgorithm:
    max_queue_size = 300
    max_estimates = 8
    max_final_iterations = 100000
    max_retries = 100

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
        self.ancestors = dict()
        self.issued = set()
        self.cached = dict()

        # Push root
        path = self.sampler.root().path.serialize()
        self.visited.add(path)
        self.pending.add(path)
        
    def serialize(self):
        # TODO: implement
        return {}

    def update(self, path, reward):
        if isinstance(path, Path):
            path = path.serialize()
        assert isinstance(path, str)

        # Update the estimate
        self.cached[path] = reward
        assert path in self.ancestors
        for ancestor_path in self.ancestors[path]:
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
        self.ancestors[path] = set()

    def sample_non_cached(self):
        # Expand the heap
        while len(self.pending) == 0 and len(self.heap) > 0:
            _, top_path = self.heap.pop()
            top_node = self.sampler.visit(Path.deserialize(top_path))
            if top_node is None:
                continue
            
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

            if node is None or node.is_dead_end():
                continue

            if node.is_final():
                self.ancestors[serialized_path] = set([serialized_path])
                self.waiting[serialized_path] = (0, set([serialized_path]))
                logging.debug(f'Found a final path: {serialized_path}')
                return [serialized_path]
            
            # Sample
            estimates = []
            for i in range(self.max_estimates):
                new_node = None
                for j in range(self.max_final_iterations):
                    node = self.sampler.random_node_with_prefix(path)
                    if node and node.is_final():
                        new_node = node
                        break
                if new_node:
                    serialized_new_path = new_node.path.serialize()
                    if serialized_new_path not in self.ancestors:
                        self.ancestors[serialized_new_path] = set()
                    self.ancestors[serialized_new_path].add(serialized_path)
                    estimates.append(serialized_new_path)
                    logging.debug(f'Found a new path: {serialized_new_path} with ancestor {serialized_path}')
            
            # Return if not empty
            if len(estimates) > 0:
                self.waiting[serialized_path] = (0, set(estimates))
                return estimates
            
        return None
    
    def sample(self):
        for i in range(self.max_retries):
            # Sample
            samples = self.sample_non_cached()
            
            if samples is None:
                continue
            
            # Update cached estimates
            to_issue = []
            for sample in samples:
                if sample in self.cached:
                    self.update(sample, self.cached[sample])
                elif sample not in self.issued:
                    self.issued.add(sample)
                    to_issue.append(sample)
            
            # Return if not empty
            if len(to_issue) > 0:
                return to_issue

        return 'end'
