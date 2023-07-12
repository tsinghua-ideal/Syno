import ctypes
import json
import logging
import os
import time
from KAS.Node import Path, Node


class Session:
    # TODO: add interface for deserializing from file
    # TODO: add timeout handling

    def __init__(self, sampler, algo, args):
        self.args = args
        self.sampler = sampler
        self.reward_power = args.kas_reward_power
        self.reward_trunc = args.kas_reward_trunc

        self.save_interval = args.kas_server_save_interval
        self.save_dir = args.kas_server_save_dir
        self.last_save_time = time.time()
        self.start_time = time.time()

        self.pending = set()
        self.waiting = set()

        # Algorithm
        # Required to implement:
        # - serialize
        # - update
        # - sample
        self.algo = algo
        assert hasattr(self.algo, 'serialize'), f'Algorithm {algo} does not implement `serialize`'
        assert hasattr(self.algo, 'update'), f'Algorithm {algo} does not implement `update`'
        assert hasattr(self.algo, 'sample'), f'Algorithm {algo} does not implement `sample`'

    def save(self, force=True):
        if self.save_dir is None:
            return

        if not force and time.time() - self.last_save_time < self.save_interval:
            return

        logging.info(f'Saving search session into {self.save_dir}')
        os.makedirs(self.save_dir, exist_ok=True)

        # Save state
        state = self.algo.serialize()
        if state:
            with open(os.path.join(self.save_dir, 'state.json'), 'w') as f:
                json.dump(state, f, indent=2)
        
        # Save arguments
        with open(os.path.join(self.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(self.args), f, indent=2)

    def update(self, path, accuracy):
        # Not more waiting
        self.waiting.remove(path)

        # Get implementation
        path = Path.deserialize(path)
        node = self.sampler.visit(path)

        # Save into directory
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            score_str = ('0' * max(0, 5 - len(f'{int(accuracy * 10000)}'))) + f'{int(accuracy * 10000)}' if accuracy >= 0 else 'ERROR'
            hash_str = f'{ctypes.c_size_t(hash(path)).value}'
            kernel_save_dir = os.path.join(self.save_dir, f'{score_str}_{hash_str}')
            os.makedirs(kernel_save_dir, exist_ok=True)

            # GraphViz
            node._node.generate_graphviz_as_final(os.path.join(kernel_save_dir, 'graph.dot'), 'kernel')
            
            # Loop
            with open(os.path.join(kernel_save_dir, 'loop.txt'), 'w') as f:
                loop_str = node._node.get_nested_loops_as_final()
                f.write(str(loop_str))

            # Meta information
            with open(os.path.join(kernel_save_dir, 'meta.json'), 'w') as f:
                json.dump({'path': path.serialize(), 'accuracy': accuracy, 'time': time.time() - self.start_time}, f, indent=2)

        # Update with reward
        reward = ((max(accuracy, self.reward_trunc) - self.reward_trunc) / (1 - self.reward_trunc)) ** self.reward_power if accuracy > 0 else -1
        self.algo.update(path, reward)

    def sample(self):
        # Get new samples if there is no pending samples
        if len(self.pending) == 0:
            new_samples = self.algo.sample()

            # String information
            if type(new_samples) == str:
                return new_samples

            # List of paths
            for new_sample in new_samples:
                assert new_sample not in self.pending
                assert new_sample not in self.waiting
                self.pending.add(new_sample)

        # Return a sample in the pending set
        assert len(self.pending) > 0
        assert len(self.waiting.intersection(self.pending)) == 0
        new_sample = self.pending.pop()
        self.waiting.add(new_sample)
        return new_sample        
