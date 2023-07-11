import logging
import sys

from .session import Session
from .mcts import MCTSAlgorithm
# TODO: add random search


def get_session(sampler, args):
    # Get algorithm
    algo_cls_name = f'{args.kas_search_algo}Algorithm'
    assert hasattr(sys.modules[__name__], algo_cls_name), f'Could not find search algorithm {args.kas_search_algo}'
    logging.info(f'Using search algorithm {args.kas_search_algo}')
    algo_cls = getattr(sys.modules[__name__], algo_cls_name)
    algo = algo_cls(sampler, args)

    # Return session
    return Session(sampler, algo, args)
