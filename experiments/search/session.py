import ctypes
import json
import logging
import os
import time
import threading
import queue
from KAS.Node import Path, Node


class Session:
    # TODO: add interface for deserializing from file
    # TODO: add timeout handling

    def __init__(self, sampler, algo, args):
        # Parameters
        self.args = args
        self.sampler = sampler
        self.reward_power: int = args.kas_reward_power
        self.reward_trunc: float = args.kas_reward_trunc
        self.target: str = args.kas_target
        self.min_accuracy: float = args.kas_min_accuracy
        self.flops_trunc: int = int(args.kas_flops_trunc)
        self.evaluate_time_limit = args.kas_evaluate_time_limit

        # Configs
        self.save_interval = args.kas_server_save_interval
        self.save_dir = args.kas_server_save_dir
        self.last_save_time = time.time()
        self.start_time = time.time()

        # Pending results
        self.pending = set()
        self.waiting = set()
        self.timeout_samples = set()

        self.time_buffer = dict()

        # Algorithm
        # Required to implement:
        # - serialize
        # - update
        # - sample
        self.algo = algo
        assert hasattr(
            self.algo, "serialize"
        ), f"Algorithm {algo} does not implement `serialize`"
        assert hasattr(
            self.algo, "update"
        ), f"Algorithm {algo} does not implement `update`"
        assert hasattr(
            self.algo, "sample"
        ), f"Algorithm {algo} does not implement `sample`"

        # Prefetcher for final node
        self.num_prefetch = args.kas_num_virtual_evaluator
        if self.num_prefetch > 0:
            self.prefetched = queue.Queue(maxsize=self.num_prefetch)
            self.prefetcher = threading.Thread(target=self.prefetcher_main)
            self.prefetcher.start()

    def save(self, force=True):
        if self.save_dir is None:
            return

        if not force and time.time() - self.last_save_time < self.save_interval:
            return

        self.last_save_time = time.time()
        logging.info(f"Saving search session into {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=True)

        # Save state
        state = self.algo.serialize()
        if state:
            with open(os.path.join(self.save_dir, "state.json"), "w") as f:
                json.dump(state, f, indent=2)

        # Save arguments
        with open(os.path.join(self.save_dir, "args.json"), "w") as f:
            json.dump(vars(self.args), f, indent=2)

    def update(self, path, accuracy, flops, params):
        # No receiving timeout kernels
        if path in self.timeout_samples:
            return

        # Not more waiting
        self.waiting.remove(path)
        self.time_buffer.pop(path)

        # Get implementation
        path = Path.deserialize(path)
        node = self.sampler.visit(path)

        # Save into directory
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            acc_str = (
                ("0" * max(0, 5 - len(f"{int(accuracy * 10000)}")))
                + f"{int(accuracy * 10000)}"
                if accuracy >= 0
                else "ERROR"
            )
            hash_str = f"{ctypes.c_size_t(hash(path)).value}"
            kernel_save_dir = os.path.join(self.save_dir, "_".join([acc_str, hash_str]))
            os.makedirs(kernel_save_dir, exist_ok=True)

            # GraphViz
            node._node.generate_graphviz_as_final(
                os.path.join(kernel_save_dir, "graph.dot"), "kernel"
            )

            # Loop
            with open(os.path.join(kernel_save_dir, "loop.txt"), "w") as f:
                f.write(str(node._node.get_nested_loops_as_final()))

            # Meta information
            with open(os.path.join(kernel_save_dir, "meta.json"), "w") as f:
                json.dump(
                    {
                        "path": path.serialize(),
                        "accuracy": accuracy,
                        "flops": flops,
                        "params": params,
                        "time": time.time() - self.start_time,
                    },
                    f,
                    indent=2,
                )

        # Update with reward
        if accuracy > 0:
            if self.target == "flops" and accuracy >= self.min_accuracy:
                reward = self.min_accuracy + max(0, 1.0 - flops / self.flops_trunc) * (
                    1 - self.min_accuracy
                )
            else:
                reward = (max(accuracy, self.reward_trunc) - self.reward_trunc) / (
                    1 - self.reward_trunc
                )
            reward = reward ** self.reward_power
        else:
            reward = -1

        logging.info(f"Updating with reward {reward} ...")
        self.algo.update(path, reward)

    def prefetcher_main(self):
        try:
            while True:
                logging.info(f"Prefetcher calls the next sample ...")
                self.prefetched.put(self.algo.sample())
        except KeyboardInterrupt:
            logging.info("Prefetcher stopped by keyboard interrupt")
            return

    def sample_impl(self):
        # No prefetch
        if self.num_prefetch == 0:
            return self.algo.sample()

        # Get a final node from the buffer
        if self.prefetched.empty():
            logging.info(f"Waiting for prefetched samples ...")
        return self.prefetched.get()

    def sample(self):
        # Get new samples if there is no pending samples
        if len(self.pending) == 0:
            self.clean_timeout_samples()
            new_samples = self.sample_impl()

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
        self.time_buffer[new_sample] = time.time()
        return new_sample

    def clean_timeout_samples(self) -> None:
        # Clean timeout samples
        timeout_samples = set()
        for sample in self.waiting:
            if time.time() - self.time_buffer[sample] > self.evaluate_time_limit:
                timeout_samples.add(sample)
        if len(timeout_samples) == 0:
            return
        logging.info(
            f"Detected timeout samples: {timeout_samples}, auto cleaning ......"
        )
        self.timeout_samples = self.timeout_samples | timeout_samples
        for sample in timeout_samples:
            self.algo.update(sample, -1)
            self.waiting.remove(sample)
            self.time_buffer.pop(sample)
