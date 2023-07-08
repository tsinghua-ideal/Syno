import itertools
import logging
import random
import requests
import time
from KAS import TreePath

from utils import models, parser, dataset


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
        logging.info(f'Post: {self.addr}/reward?path={path}?reward={reward}')
        self.session.post(f'{self.addr}/reward?path={path}?reward={reward}')


if __name__ == '__main__':
    args = parser.arg_parse()

    logging.info('Preparing model ...')
    model, sampler = models.get_model(args, return_sampler=True)

    logging.info('Loading dataset ...')
    train_loader, val_loader = dataset.get_dataloader(args)

    logging.info('Starting server ...')
    client_handler = Handler(args)

    logging.info('Starting search ...')
    round_range = range(args.kas_search_rounds) if args.kas_search_rounds > 0 else itertools.count()

    for round in round_range:
        # Request a new kernel
        logging.info('Requesting a new kernel ...')
        while True:
            path = client_handler.sample()
            if path == 'retry':
                logging.info('No path returned, retrying ...')
                time.sleep(args.kas_retry_interval)
                continue
            break

        if path == 'end':
            logging.info('Exhausted search space, exiting ...')
            break

        path_impl = TreePath.deserialize(path)
        logging.info(f'Got a new kernel: {path_impl}')

        # Mock evaluate
        if args.kas_mock_evaluate:
            logging.info('Mock evaluating ...')
            if random.random() < 0.5:
                client_handler.success(path, random.random())
            else:
                client_handler.failure(path)
        else:
            raise NotImplementedError
