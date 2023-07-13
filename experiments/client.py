import itertools
import logging
import random
import requests
import time
from KAS import Path

from base import log, models, parser, dataset, trainer


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
                client.reward(path, -1 if random.random() < 0.5 else random.random())
                continue
            
            # Evaluate on a dataset
            kernel_packs = sampler.realize(model, node).construct_kernel_packs()
            sampler.replace(model, kernel_packs)

            logging.info('Evaluating on real dataset ...')
            flops, params = model.profile()
            logging.debug("Model flops: {:.2f}G, params: {:.2f}M".format(flops / 1e9, params / 1e6))
            _, val_errors = trainer.train(model, train_dataloader, val_dataloader, args)
            accuracy = 1 - min(val_errors)
            if args.kas_min_accuracy > accuracy or args.kas_max_flops < flops:
                client.reward(path, -1)
            else:
                if args.kas_target == 'accuracy':
                    client.reward(path, accuracy)
                elif args.kas_target == 'flops':
                    client.reward(path, 1. - flops / args.kas_max_flops)
                else:
                    raise ValueError(f'Unknown target: {args.kas_target}')
    except KeyboardInterrupt:
        logging.info('Interrupted by user, exiting ...')
