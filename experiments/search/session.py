import ctypes
import json
import logging
import os, shutil
import time
import threading
import queue
import traceback
from KAS import KernelLoader, Node, Path, Sampler, Statistics, SearchSpaceExplorer


class Session:
    # TODO: add interface for deserializing from file

    def __init__(self, sampler: Sampler, model, algo, args):
        # Parameters
        self.args = args
        self.sampler = sampler
        self.model = model
        self.reward_power: int = args.kas_reward_power
        self.reward_lower_bound: float = args.kas_acc_lower_bound
        self.reward_upper_bound: float = args.kas_acc_upper_bound
        self.target: str = args.kas_target
        self.min_accuracy: float = args.kas_min_accuracy
        self.original_flops: int = args.original_flops
        self.max_flops_ratio: float = args.kas_max_flops_ratio
        self.evaluate_time_limit = args.kas_evaluate_time_limit
        # Also the limit for the first minute training
        self.max_loss = args.gpt_max_loss

        # Configs
        self.save_interval = args.kas_server_save_interval
        self.stats_interval = args.kas_stats_interval
        self.save_dir = args.kas_server_save_dir
        self.cache_dir = args.kas_send_cache_dir
        self.evaluation_result_file = args.kas_node_cache_dirs
        self.last_save_time = time.time()
        self.last_stats_time = time.time()
        self.start_time = time.time()

        # Pending results
        self.pending = set()
        self.waiting = set()
        self.timeout_samples = set()

        self.time_buffer = dict()

        # Algorithm
        # Required to implement:
        # - serialize
        # - deserialize
        # - update
        # - sample
        self.algo = algo
        assert hasattr(
            self.algo, "serialize"
        ), f"Algorithm {algo} does not implement `serialize`"
        assert hasattr(
            self.algo, "deserialize"
        ), f"Algorithm {algo} does not implement `deserialize`"
        assert hasattr(
            self.algo, "update"
        ), f"Algorithm {algo} does not implement `update`"
        assert hasattr(
            self.algo, "sample"
        ), f"Algorithm {algo} does not implement `sample`"

        # Prefetcher for final node
        self.num_prefetch = args.kas_num_virtual_evaluator

        self.search_space_explorer = SearchSpaceExplorer(model, sampler)

    def start_prefetcher(self):
        if self.num_prefetch > 0:
            self.prefetched = queue.Queue(maxsize=self.num_prefetch)
            self.prefetcher = threading.Thread(target=self.prefetcher_main)
            self.prefetcher.start()

    def print_stats(self, force=True):
        if not force and time.time() - self.last_stats_time < self.stats_interval:
            return

        Statistics.PrintLog(self.sampler)
        self.last_stats_time = time.time()

    def save(self, force=True):
        if self.save_dir is None:
            return

        if not force and time.time() - self.last_save_time < self.save_interval:
            return

        logging.info(f"Saving search session into {self.save_dir}")
        os.makedirs(self.save_dir, exist_ok=True)

        try:
            # Save state
            state = self.algo.serialize()
            if state:
                with open(os.path.join(self.save_dir, "state.json"), "w") as f:
                    json.dump(state, f, indent=2)

            # Save arguments
            with open(os.path.join(self.save_dir, "args.json"), "w") as f:
                json.dump(vars(self.args), f, indent=2)
        except Exception as e:
            logging.info(f"Saving failed. {e} {traceback.format_exc()}")

        self.last_save_time = time.time()

    def fast_update(self):
        if not os.path.exists(self.evaluation_result_file):
            return

        logging.info(f"Fast updating with files in {self.evaluation_result_file}")

        with open(self.evaluation_result_file) as f:
            dirs = [l[:-1] for l in f.readlines() if l[-1] == "\n"]

        kernels = []
        for directory in dirs:
            if not os.path.exists(directory):
                logging.warning(f"{directory} does not exist......")
                continue
            for kernel_fmt in os.listdir(directory):
                kernel_dir = os.path.join(directory, kernel_fmt)
                if not os.path.isdir(kernel_dir):
                    continue
                if "ERROR" in kernel_dir:
                    continue
                if "cache" in kernel_dir:
                    continue
                files = list(os.listdir(kernel_dir))
                assert (
                    "graph.dot" in files
                    and "loop.txt" in files
                    and "meta.json" in files
                )

                meta_path = os.path.join(kernel_dir, "meta.json")
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                kernels.append(
                    (
                        meta["time"],
                        meta["path"],
                        meta["accuracy"],
                        meta["loss"],
                        meta["flops"],
                        meta["params"],
                    )
                )
        kernels = sorted(kernels, key=lambda x: x[0])

        for kernel in kernels:
            logging.info(f"Fast updating with {kernel[1]}")
            path, accuracy, loss, flops, params = (
                kernel[1],
                kernel[2],
                kernel[3],
                kernel[4],
                kernel[5],
            )
            # Update with reward
            if self.target == "loss":
                reward = max((self.max_loss - loss), 0) / self.max_loss
                reward = reward ** self.reward_power
                if loss >= self.max_loss:
                    reward = -1
            elif accuracy > 0:
                if self.target == "flops" and accuracy >= self.min_accuracy:
                    reward = self.min_accuracy + max(
                        0, 1.0 - (flops / self.original_flops) / self.max_flops_ratio
                    ) * (1 - self.min_accuracy)
                else:
                    reward = (
                        (
                            max(accuracy, self.reward_lower_bound)
                            - self.reward_lower_bound
                        )
                        / (1 - self.reward_lower_bound)
                        / self.reward_upper_bound
                    )
                reward = reward ** self.reward_power
            else:
                reward = -1

            self.algo.load_eval_result(path, reward)

    def load(self):
        if not os.path.exists(self.save_dir):
            logging.info("Specified path does not exists, skip loading session. ")
            return

        # load state
        state = json.load(open(os.path.join(self.save_dir, "state.json")))
        self.algo.deserialize(state)
        logging.info("Successfully loaded session. ")

    def update(self, path, accuracy, flops, params, kernel_flag, loss):
        # No receiving timeout kernels
        if path in self.timeout_samples:
            logging.debug(f"{path} is removed due to timeout...")
            return
        if path not in self.waiting:
            logging.warning(f"{path} is not in our waiting queue")
            return

        # Not more waiting
        self.waiting.remove(path)
        self.time_buffer.pop(path)

        # Get implementation
        path = Path.deserialize(path)
        node = self.sampler.visit(path)

        if self.target == "loss":
            assert loss is not None
        else:
            loss = 0

        # Save into directory
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            if self.target == "loss":
                acc_str = (
                    ("0" * max(0, 5 - len(f"{int(loss * 1000)}")))
                    + f"{int(loss * 1000)}"
                    if loss > 0
                    else "ERROR"
                )
            else:
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
                        "kernel_flag": kernel_flag,
                        "time": time.time() - self.start_time,
                        "loss": loss,
                    },
                    f,
                    indent=2,
                )

            # copying kernel dir
            if kernel_flag != "MOCKPATH":
                shutil.copytree(
                    self.sampler.realize(self.model, node).get_directory(),
                    os.path.join(kernel_save_dir, "kernel_scheduler_dir"),
                )

        # Update with reward
        if self.target == "loss":
            reward = max((self.max_loss - loss), 0) / self.max_loss
            reward = reward ** self.reward_power
            if loss >= self.max_loss:
                reward = -1
        elif accuracy > 0:
            if self.target == "flops" and accuracy >= self.min_accuracy:
                reward = self.min_accuracy + max(
                    0, 1.0 - (flops / self.original_flops) / self.max_flops_ratio
                ) * (1 - self.min_accuracy)
            else:
                reward = (
                    (max(accuracy, self.reward_lower_bound) - self.reward_lower_bound)
                    / (1 - self.reward_lower_bound)
                    / self.reward_upper_bound
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
                try:
                    sample = self.algo.sample()
                    time.sleep(1)
                except Exception as e:
                    logging.info(
                        f"Prefetcher encounters error {e}. {traceback.format_exc()}"
                    )
                    continue
                self.prefetched.put(sample)
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

    def path_to_file(self, path: str) -> str:
        node = self.sampler.visit(Path.deserialize(path))
        kernel = self.sampler.realize(self.model, node)
        directory = kernel.get_directory()
        working_dir = os.path.join(self.cache_dir, os.path.basename(directory))
        file_name = os.path.join(working_dir, "kernel.tar.gz")
        kernel.archive_to(file_name, overwrite=False)
        return file_name

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
            path = Path.deserialize(sample)
            self.algo.update(path, -1)
            self.waiting.remove(sample)
            self.time_buffer.pop(sample)
