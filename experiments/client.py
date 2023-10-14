import itertools
import logging
import random
import requests
import time
import os, sys, shutil
import tarfile
from KAS import Path
from typing import List, Dict, Tuple, Union

from base import log, models, parser, dataset, trainer, mem


class Handler:
    timeout = 600

    def __init__(self, args):
        self.addr = f"http://{args.kas_server_addr}:{args.kas_server_port}"
        self.session = requests.Session()
        # Allow localhost
        self.session.trust_env = False
        self.cache_dir = args.kas_client_cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def sample(self) -> str:
        logging.info(f"Get: {self.addr}/sample")
        j = self.session.get(f"{self.addr}/sample", timeout=self.timeout).json()
        assert "path" in j
        return j["path"]
    
    def fetch(self, path: str) -> str:
        logging.info(f"Fetch: {self.addr}/fetch/{path}")
        response = self.session.get(f"{self.addr}/fetch/{path}", timeout=self.timeout)
        file_name = os.path.join(self.cache_dir, "kernel.tar.gz")
        folder_name = os.path.join(self.cache_dir, "kernel_dir")
        if os.path.exists(file_name):
            os.remove(file_name)
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        with open(file_name, mode='wb') as f:               
            f.write(response.content)
        with tarfile.open(file_name, 'r') as tar:
            tar.extractall(self.cache_dir)
        assert os.path.exists(folder_name)
        return folder_name

    def reward(self, path, accuracy, flops, params, kernel_dir):
        message = f"{self.addr}/reward?path={path}&accuracy={accuracy}&flops={flops}&params={params}&kernel_dir={kernel_dir}"
        logging.info("Post: " + message)
        self.session.post(message, timeout=self.timeout)


def main():

    logging.info("Preparing model ...")
    model = models.get_model(args, return_sampler=False)

    logging.info("Loading dataset ...")
    train_dataloader, val_dataloader = dataset.get_dataloader(args)

    logging.info("Starting server ...")
    client = Handler(args)

    logging.info("Starting search ...")
    round_range = (
        range(args.kas_search_rounds)
        if args.kas_search_rounds > 0
        else itertools.count()
    )

    try:
        for round in round_range:
            # Request a new kernel
            logging.info("Requesting a new kernel ...")
            while True:
                path = client.sample()
                if path == "retry":
                    logging.info(
                        f"No path returned, retrying in {args.kas_retry_interval} second(s) ..."
                    )
                    time.sleep(args.kas_retry_interval)
                    continue
                break

            if path == "end":
                logging.info("Exhausted search space, exiting ...")
                break

            logging.info(f"Got a new path: {path}")
            
            while True:
                try:
                    kernel_directory = client.fetch(path)
                    logging.debug(f"Fetched {path} in {kernel_directory}. ")
                    break
                except Exception as e:
                    logging.debug(f"Fetching {path} failed because of {e}. Retrying......")

            # Mock evaluate
            if args.kas_mock_evaluate:
                logging.info("Mock evaluating ...")
                time.sleep(10)
                client.reward(
                    path,
                    -1 if random.random() < 0.5 else random.random(),
                    random.randint(int(1e6), int(1e7)),
                    random.randint(int(1e6), int(1e7)),
                    "MOCKPATH"
                )
                continue

            # Load and evaluate on a dataset
            try:
                try:
                    kernel_dir = model.load_kernel(kernel_directory, compile=args.compile, batch_size=args.batch_size, seq_len=args.gpt_seq_len)
                except Exception as e:
                    if args.compile:
                        logging.warning("torch compile error, falling back to non-compile version. ")
                        kernel_dir = model.load_kernel(kernel_directory, compile=False, batch_size=args.batch_size, seq_len=args.gpt_seq_len)
                    else:
                        raise e
                flops, params = model.profile(args.batch_size, seq_len=args.gpt_seq_len)
                logging.debug(
                    f"Loaded model has {flops} FLOPs per batch and {params} parameters in total."
                )

                logging.info("Evaluating on real dataset ...")
                if "gpt" not in args.model:
                    accuracy = max(trainer.train(model, train_dataloader, val_dataloader, args))
                else:
                    accuracy = (5 - trainer.train_gpt(model, train_dataloader, val_dataloader, args)[-1][1]) / 5
                    if accuracy <= 0:
                        accuracy = -1
            except Exception as e:
                if not "out of memory" in str(e):
                    raise e
                logging.warning(f"OOM when evaluating {path}, skipping ...")
                model.remove_thop_hooks()
                flops, params, accuracy, kernel_dir = 0, 0, -1, "EMPTY"
            client.reward(path, accuracy, flops, params, kernel_dir)
    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting ...")


if __name__ == "__main__":
    log.setup()

    args = parser.arg_parse()

    # Set memory limit
    mem.memory_limit(args.client_mem_limit)

    try:
        main()
    except MemoryError:
        sys.stderr.write("\n\nERROR: Memory Exception\n")
        sys.exit(1)
