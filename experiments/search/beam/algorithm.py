import logging
import random
from typing import List, Tuple, Dict
from KAS.Sampler import Sampler
from KAS.Node import Path


class BeamAlgorithm:
    max_queue_size = 1000
    max_final_iterations = 150
    expand_async_layers = 3
    noise_weight = 0.4

    def __init__(self, sampler: Sampler, args):
        self.sampler = sampler

        # Heap
        self.heap = []
        # max_reward, to_issue_ests
        self.score: Dict[str, Tuple[float, int]] = dict()

        # For estimating
        self.max_depth = args.kas_depth
        self.ancestors = dict()
        self.issued = set()
        self.cached = dict()

        # Push root
        self.push_heap(self.sampler.root().path)
        
    def serialize(self):
        return {'heap': self.heap, 'score': self.score, 'cached': self.cached}
    
    def sort_heap(self):
        # With noise
        self.heap = sorted(self.heap, key=lambda x: self.score[x][0] * (1 - self.noise_weight) + random.random() * self.noise_weight)

    def push_heap(self, path: Path):
        if path is None:
            return

        # Already visited
        serialized_path = path.serialize()
        if serialized_path in self.score:
            return
        
        # Not visited
        node = self.sampler.visit(path)
        node.expand_async(self.expand_async_layers)
        num_est = self.max_depth - len(path) + 2
        assert num_est > 0
        logging.info(f'Pushing path({serialized_path}, depth: {len(path)}, est: {num_est}) to heap with async expand ...')
        self.score[serialized_path] = (0, num_est)
        self.heap.append(serialized_path)
        self.sort_heap()
        if len(self.heap) > self.max_queue_size:
            self.heap.pop(0)
        assert len(self.heap) <= self.max_queue_size

    def update(self, path, reward):
        if isinstance(path, Path):
            path = path.serialize()
        assert isinstance(path, str)

        # Update the estimate
        self.cached[path] = reward
        assert path in self.ancestors, f'Path({path}) not in ancestors'
        for ancestor_path in self.ancestors[path]:
            assert ancestor_path in self.score
            max_reward, issued_ests = self.score[ancestor_path]
            self.score[ancestor_path] = (max(max_reward, reward), issued_ests)

        # Maintain the heap
        self.sort_heap()
        del self.ancestors[path]

    def get_all_ests_to_issue(self, serialized_path: str):
        path = Path.deserialize(serialized_path)
        node = self.sampler.visit(path)
        if node is None:
            return set()

        estimates = set()
        if node.is_final():
            estimates.add(serialized_path)
        else:
            for _ in range(self.score[serialized_path][1]):
                new_node = None
                for i in range(self.max_final_iterations):
                    final_node = self.sampler.random_node_with_prefix(path)
                    if final_node and final_node.is_final() and not final_node.is_dead_end():
                        new_node = final_node
                        logging.info(f'Got final node at hit {i} / {self.max_final_iterations}')
                        break
                if new_node:
                    estimates.add(new_node.path.serialize())
        return estimates


    def sample_non_cached(self):
        while len(self.heap) > 0:
            # Check whether need more to issue
            for path in reversed(self.heap):
                if self.score[path][1] > 0:
                    # Get estimates
                    logging.info(f'Need more to issue: Path({path})')
                    estimates = self.get_all_ests_to_issue(path)
                    self.score[path] = (self.score[path][0], 0)

                    # Set ancestors
                    for est in estimates:
                        if est not in self.ancestors:
                            self.ancestors[est] = set()
                        self.ancestors[est].add(path)
                    if len(estimates) == 0:
                        continue
                    logging.info(f'Got estimates: {estimates}')
                    return estimates
                
            # No need to issue, just expand the top
            logging.info(f'No need to issue, expand the top ...')
            path = self.heap.pop(-1)
            node = self.sampler.visit(Path.deserialize(path))
            if node is None or node.is_dead_end():
                logging.info(f'Top path({path}) is dead end, skipping ...')
                continue
            
            for handle in node.get_children_handles():
                if node.is_dead_end():
                    break

                child = node.get_child(handle)
                if child.is_dead_end():
                    continue

                self.push_heap(child.path)
        
        # Exhausted
        return None
    
    def sample(self):
        while True:
            # Sample
            samples = self.sample_non_cached()
            
            if samples is None:
                return 'end'
            
            # Update cached estimates
            to_issue = []
            for sample in samples:
                if sample in self.cached:
                    logging.info(f'Cached sample: {sample}')
                    self.update(sample, self.cached[sample])
                elif sample not in self.issued:
                    self.issued.add(sample)
                    to_issue.append(sample)
            
            # Return if not empty
            if len(to_issue) > 0:
                return to_issue
