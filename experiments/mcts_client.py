import itertools
import logging
import random
import requests
import time
from KAS import TreePath

from utils import log, models, parser, dataset, trainer


class Handler:
    def __init__(self, args):
        self.addr = f'http://{args.kas_server_addr}:{args.kas_server_port}'
        self.session = requests.Session()
        # Allow localhost
        self.session.trust_env = False

    def sample(self):
        logging.info(f'Get: {self.addr}/sample')
        j = self.session.get(f'{self.addr}/sample').json()
        assert 'path' in j
        return j['path']

    def reward(self, path, reward):
        logging.info(f'Post: {self.addr}/reward?path={path}&value={reward}')
        self.session.post(f'{self.addr}/reward?path={path}&value={reward}')


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
        
        node = sampler.visit(TreePath.deserialize(path))
        logging.info(f'Got a new kernel: {node}')

        # Mock evaluate
        if args.kas_mock_evaluate:
            logging.info('Mock evaluating ...')
            time.sleep(1)
            client.reward(path, -1 if random.random() < 0.5 else random.random())
            continue
        
        # Evaluate on a dataset
        kernel_packs = sampler.realize(model, node).construct_kernel_packs()
        sampler.replace(model, kernel_packs)

        logging.info('Evaluating on real dataset ...')
        _, val_errors = trainer.train(model, train_dataloader, val_dataloader, args)
        accuracy = 1 - min(val_errors)
        client.reward(path, accuracy)
