import itertools
import logging
import random
import requests
import time
import os, sys, shutil
import tarfile
import numpy as np
from KAS import KernelLoader
from requests.exceptions import ConnectionError

from base import log, models, parser, dataset, trainer, mem


class Handler:
    timeout = None

    def __init__(self, args):
        self.addr = f"http://{args.kas_server_addr}:{args.kas_server_port}"
        self.session = requests.Session()
        # Allow localhost
        self.session.trust_env = False
        self.cache_dir = args.kas_client_cache_dir
        self.kernel_buffer = os.path.join(self.cache_dir, "evaluating_kernel.txt")
        os.makedirs(self.cache_dir, exist_ok=True)

    def sample(self) -> str:
        logging.info(f"Get: {self.addr}/sample")
        j = self.session.get(f"{self.addr}/sample", timeout=self.timeout).json()
        assert "path" in j
        return j["path"]

    def fetch(self, path: str) -> str | None:
        logging.info(f"Fetch: {self.addr}/fetch/{path}")
        response = self.session.get(f"{self.addr}/fetch/{path}", timeout=self.timeout)
        try:
            exc = response.json()["Exception"]
            logging.info(f"Fetch failed because of {exc}")
            return None
        except requests.exceptions.JSONDecodeError:
            pass
        file_name = os.path.join(self.cache_dir, "kernel.tar.gz")
        folder_name = os.path.join(self.cache_dir, "kernel_dir")
        if os.path.exists(file_name):
            os.remove(file_name)
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        with open(file_name, mode="wb") as f:
            f.write(response.content)
        with tarfile.open(file_name, "r") as tar:
            tar.extractall(self.cache_dir)
        assert os.path.exists(folder_name)
        return folder_name

    def reward(self, path, accuracy, flops, params, kernel_flag, loss):
        message = f"{self.addr}/reward?path={path}&accuracy={accuracy}&flops={flops}&params={params}&kernel_flag={kernel_flag}&loss={loss}"
        logging.info("Post: " + message)
        while True:
            try:
                self.session.post(message, timeout=self.timeout)
                return
            except:
                pass


def main():
    sample_input = None
    logging.info("Loading dataset ...")
    train_dataloader, val_dataloader = dataset.get_dataloader(args)
    if "gcn" in args.model:
        sample_input = train_dataloader

    logging.info("Preparing model ...")
    model = models.get_model(args, return_sampler=False, sample_input=sample_input)

    logging.info("Starting server ...")
    client = Handler(args)

    if os.path.exists(client.kernel_buffer):
        path = open(client.kernel_buffer).read()
        logging.info(f"Detected unfinished kernel {path}")
        flops, params, accuracy, kernel_flag, loss = (
            0,
            0,
            -1,
            "EMPTY",
            args.gpt_max_loss,
        )
        client.reward(path, accuracy, flops, params, kernel_flag, loss)
        os.remove(client.kernel_buffer)

    logging.info("Starting search ...")
    round_range = (
        range(args.kas_search_rounds)
        if args.kas_search_rounds > 0
        else itertools.count()
    )

    for _ in round_range:
        # Request a new kernel
        logging.info("Requesting a new kernel ...")
        while True:
            try:
                path = client.sample()
                if path == "retry":
                    logging.info(
                        f"No path returned, retrying in {args.kas_retry_interval} second(s) ..."
                    )
                    time.sleep(args.kas_retry_interval)
                    continue
                break
            except ConnectionError as e:
                logging.info(f"Sample failed because of {e}, retrying")
                time.sleep(10)

        if path == "end":
            logging.info("Exhausted search space, exiting ...")
            break

        logging.info(f"Got a new path: {path}")

        with open(client.kernel_buffer, "w") as f:
            f.write(path)

        while True:
            try:
                kernel_directory = client.fetch(path)
                logging.info(f"Fetched {path} in {kernel_directory}. ")
                break
            except Exception as e:
                logging.info(f"Fetching {path} failed because of {e}. Retrying......")
                time.sleep(10)

        if kernel_directory is None:
            continue

        # Mock evaluate
        if args.kas_mock_evaluate:
            logging.info("Mock evaluating ...")
            time.sleep(10)
            client.reward(
                path,
                -1 if random.random() < 0.5 else random.random(),
                random.randint(int(1e6), int(1e7)),
                random.randint(int(1e6), int(1e7)),
                "MOCKPATH",
                0,
            )
            continue

        # Load and evaluate on a dataset
        kernel_loader = KernelLoader.from_directory(kernel_directory)
        kernel_flag = model.load_kernel(
            kernel_loader,
            compile=args.compile,
            batch_size=args.batch_size,
            seq_len=args.gpt_seq_len,
        )
        flops, params = model.profile(args.batch_size, seq_len=args.gpt_seq_len)
        logging.info(
            f"Loaded model has {flops} FLOPs per batch and {params} parameters in total."
        )

        logging.info("Evaluating on real dataset ...")
        if "gpt" not in args.model:
            accuracy = max(trainer.train(model, train_dataloader, val_dataloader, args))
            loss = 0
        else:
            accuracy = 0
            losses = trainer.train_gpt(model, train_dataloader, val_dataloader, args)
            losses = list(map(lambda t: t[1], losses))
            assert len(losses) >= 1
            len_not_avg = max(int(len(losses) * 0.8), 1)
            loss = np.mean(losses[len_not_avg - 1 :])
            logging.info(f"Meaned loss of last 20%: {loss}")

        client.reward(path, accuracy, flops, params, kernel_flag, loss)
        os.remove(client.kernel_buffer)


if __name__ == "__main__":
    log.setup(logging.INFO)

    args = parser.arg_parse()

    # Set memory limit
    mem.memory_limit(args.client_mem_limit)

    main()
