import itertools
import logging
import random
import requests
import time
import sys
from KAS import Path

from base import log, models, parser, dataset, trainer, mem


class Handler:
    timeout = 600

    def __init__(self, args):
        self.addr = f"http://{args.kas_server_addr}:{args.kas_server_port}"
        self.session = requests.Session()
        # Allow localhost
        self.session.trust_env = False

    def sample(self):
        logging.info(f"Get: {self.addr}/sample")
        j = self.session.get(f"{self.addr}/sample", timeout=self.timeout).json()
        assert "path" in j
        return j["path"]

    def reward(self, path, accuracy, flops, params, kernel_dir):
        message = f"{self.addr}/reward?path={path}&accuracy={accuracy}&flops={flops}&params={params}&kernel_dir={kernel_dir}"
        logging.info("Post: " + message)
        self.session.post(message, timeout=self.timeout)


def main():

    logging.info("Preparing model ...")
    model, sampler = models.get_model(args, return_sampler=True)

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
            node = sampler.visit(Path.deserialize(path))

            # Mock evaluate
            if args.kas_mock_evaluate:
                logging.info("Mock evaluating ...")
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
                    kernel_dir = model.load_kernel(
                        sampler, node, compile=args.compile, batch_size=args.batch_size
                    )
                except Exception as e:
                    if args.compile:
                        logging.warning("torch compile error, falling back to non-compile version. ")
                        kernel_dir = model.load_kernel(
                            sampler, node, compile=False, batch_size=args.batch_size
                        )
                    else:
                        raise e
                flops, params = model.profile(args.batch_size, args=args)
                logging.debug(
                    f"Loaded model has {flops} FLOPs per batch and {params} parameters in total."
                )

                logging.info("Evaluating on real dataset ...")
                accuracy = max(
                    trainer.train(model, train_dataloader, val_dataloader, args)
                )
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
