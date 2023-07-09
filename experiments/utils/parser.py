import argparse
import logging
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .models import get_model_input_size


def arg_parse():
    parser = argparse.ArgumentParser(description='KAS trainer/searcher')

    # Model
    parser.add_argument('--model', type=str, default='FCNet')

    # Dataset
    parser.add_argument('--dataset', type=str, default='torch/mnist')
    parser.add_argument('--root', type=str, default='~/data')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='Random seed (default: 42)')
    parser.add_argument('--mean', type=float, nargs='+', default=IMAGENET_DEFAULT_MEAN, metavar='MEAN',
                        help='Mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=IMAGENET_DEFAULT_STD, metavar='STD',
                        help='Variance of pixel value of dataset')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', 
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2, metavar='N',
                        help='How many training processes to use')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=True,
                        help='Use the multi-epochs-loader to save time at the beginning of every epoch')
    parser.add_argument('--fetch-all-to-gpu', action='store_true', default=False,
                        help='Fetch all data to GPU before training')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='Weight decay (default: 0.05)')

    # Scheduler parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler')
    parser.add_argument('--warmup-lr', type=float, default=0.01, metavar='LR',
                        help='Warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='Number of epochs to train (default: 300)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                        help='Epochs to cooldown LR at min_lr, after cyclic schedule ends')
    
    # KAS preferences
    parser.add_argument('--kas-replace-placeholder', type=str, default=None)
    parser.add_argument('--kas-depth', default=4,
                        type=int, help='KAS sampler depth')
    parser.add_argument('--kas-max-tensors', default=2,
                        type=int, help='KAS sampler maximum tensors')
    parser.add_argument('--kas-max-reductions', default=2,
                        type=int, help='KAS sampler maximum reductions')
    parser.add_argument('--kas-min-dim', default=2,
                        type=int, help='KAS sampler minimum dimensions')
    parser.add_argument('--kas-max-dim', default=4,
                        type=int, help='KAS sampler maximum dimensions')
    parser.add_argument('--kas-scheduler-cache-dir', default='.scheduler-cache',
                        help='KAS sampler saving directory')
    parser.add_argument('--kas-server-save-interval', default=600, type=int, help='KAS server saving interval (in seconds)')
    parser.add_argument('--kas-server-save-dir', default=None, type=str, help='KAS server saving directory')
    parser.add_argument('--kas-max-flops', default=1e15, type=float,
                        help='Maximum FLOPs for searched kernels (in G-unit, only for search)')
    parser.add_argument('--kas-reward-power', default=2, type=float, help='Reward power')
    
    # MCTS preferences
    parser.add_argument('--kas-server-addr', default='localhost', type=str, help='MCTS server address')
    parser.add_argument('--kas-server-port', default=8000, type=int, help='MCTS server port')
    parser.add_argument('--kas-search-rounds', default=0, type=int, help='MCTS rounds')
    parser.add_argument('--kas-mock-evaluate', action='store_true', default=False, help='Mock evaluate')
    parser.add_argument('--kas-retry-interval', default=10, type=float, help='Client retry time interval')

    args = parser.parse_args()

    # Extra arguments
    setattr(args, 'input_size', get_model_input_size(args))

    # Print
    args_str = "\n  > ".join([f"{k}: {v}" for k, v in vars(args).items()])
    logging.info(f'Execution arguments: \n  > {args_str}')

    return args
