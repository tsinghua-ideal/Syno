import os
import torch
import logging

from timm.utils import random_seed


def initialize(args):
    assert torch.cuda.is_available(), 'No available CUDA devices'

    # Distributed.
    setattr(args, 'distributed', False)
    setattr(args, 'world_size', 1)
    setattr(args, 'rank', 0)
    setattr(args, 'device', 'cuda:0')
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        logging.info('Training in distributed mode with multiple processes, 1 GPU per process. Process {}, total {}.'
                     .format(args.rank, args.world_size))
    else:
        logging.info('Training with a single process on 1 GPU.')
    assert args.rank >= 0

    # Set seed.
    random_seed(args.seed, args.rank)

    return args
