import itertools
import logging
import random
import requests
from KAS import TreePath

from utils import models, parser, dataset


class Client:
    def __init__(self, args):
        self.addr = f'{args.kas_server_addr}:{self.kas_server_port}'

    def sample(self):
        j = requests.get(f'{self.addr}/sample').json()
        return j['path'] if 'path' in j else None

    def reward(self, path, reward):
        print(f'Posting: {self.addr}/reward?path={path}?reward={reward}')
        requests.post(f'{self.addr}/reward?path={path}?reward={reward}')


if __name__ == '__main__':
    args = parser.arg_parse()

    logging.info('Preparing model ...')
    model, _ = models.get_model_and_sampler(args)

    logging.info('Loading dataset ...')
    train_loader, val_loader = dataset.get_dataloader(args)

    logging.info('Starting server ...')
    client_handler = Client(args)

    logging.info('Starting search ...')
    round_range = range(args.kas_search_rounds) if args.kas_search_rounds > 0 else itertools.count()

    for round in round_range:
        # Request a new kernel
        logging.info(' > Requesting a new kernel ...')
        path = client_handler.get_path()
        if path is None:
            logging.info(' > No more kernel to evaluate, exiting ...')
        path_impl = TreePath.deserialize(path)
        logging.info(f' > Got a new kernel: {path_impl}')

        # Mock evaluate
        if args.kas_mock_evaluate:
            logging.info(' > Mock evaluating ...')
            if random.random() < 0.5:
                client_handler.success(path, random.random())
            else:
                client_handler.failure(path)
        else:
            raise NotImplementedError
