import itertools
import logging
import random
import requests
import time
from KAS import Path

from base import log, models, parser, dataset, trainer


class Handler:
    timeout = 600

    def __init__(self, args):
        self.addr = f'http://{args.kas_server_addr}:{args.kas_server_port}'
        self.session = requests.Session()
        # Allow localhost
        self.session.trust_env = False

    def sample(self):
        logging.info(f'Get: {self.addr}/sample')
        j = self.session.get(f'{self.addr}/sample', timeout=self.timeout).json()
        assert 'path' in j
        return j['path']

    def reward(self, path, accuracy, flops, params):
        logging.info(f'Post: {self.addr}/reward?path={path}&accuracy={accuracy}&flops={flops}&params={params}')
        self.session.post(f'{self.addr}/reward?path={path}&accuracy={accuracy}&flops={flops}&params={params}', timeout=self.timeout)


if __name__ == '__main__':
    log.setup()

    args = parser.arg_parse()

    logging.info('Preparing model ...')
    model, sampler = models.get_model(args, return_sampler=True)

    logging.info('Loading dataset ...')
    train_dataloader, val_dataloader = dataset.get_dataloader(args)

    logging.info('Starting server ...')
    client = Handler(args)

    logging.info('Starting search ...')
    round_range = range(args.kas_search_rounds) if args.kas_search_rounds > 0 else itertools.count()

    try:
        for round in round_range:
            # Request a new kernel
            logging.info('Requesting a new kernel ...')
            while True:
                path = client.sample()
                if path == 'retry':
                    logging.info(f'No path returned, retrying in {args.kas_retry_interval} second(s) ...')
                    time.sleep(args.kas_retry_interval)
                    continue
                break

            if path == 'end':
                logging.info('Exhausted search space, exiting ...')
                break
            
            logging.info(f'Got a new path: {path}')
            node = sampler.visit(Path.deserialize(path))

            # Mock evaluate
            if args.kas_mock_evaluate:
                logging.info('Mock evaluating ...')
                client.reward(path, -1 if random.random() < 0.5 else random.random(), random.randint(int(1e6), int(1e7)), random.randint(int(1e6), int(1e7)))
                continue
            
            # Load and evaluate on a dataset
            try:
                model.load_kernel(sampler, node, compile=args.compile, batch_size=args.batch_size)
                flops, params = model.profile(args.batch_size)
                logging.debug(f"Loaded model has {flops} FLOPs per batch and {params} parameters in total.")

                logging.info('Evaluating on real dataset ...')
                accuracy = max(trainer.train(model, train_dataloader, val_dataloader, args))
            except Exception as e:
                if not "out of memory" in str(e):
                    raise e
                logging.warning(f"OOM when evaluating {path}, skipping ...")
                model.remove_thop_hooks()
                flops, params, accuracy = 0, 0, -1
            client.reward(path, accuracy, flops, params)
    except KeyboardInterrupt:
        logging.info('Interrupted by user, exiting ...')
