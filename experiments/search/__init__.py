import logging
import sys

from .session import Session
from .beam import BeamAlgorithm
from .mcts import MCTSAlgorithm, MCTSExplorer, MCTSTree
from .random import RandomAlgorithm


def get_session(sampler, model, args):
    # Get algorithm
    algo_cls_name = f"{args.kas_search_algo}Algorithm"
    assert hasattr(
        sys.modules[__name__], algo_cls_name
    ), f"Could not find search algorithm {args.kas_search_algo}"
    logging.info(f"Using search algorithm {args.kas_search_algo}")
    algo_cls = getattr(sys.modules[__name__], algo_cls_name)
    algo = algo_cls(sampler, model, args)

    session = Session(sampler, model, algo, args)
    if args.kas_resume:
        session.load()
    session.fast_update()
    session.start_prefetcher()

    # Return session
    return session
