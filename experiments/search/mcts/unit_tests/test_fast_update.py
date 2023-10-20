"""
Not finished test. 
"""

import os, sys, json
import logging
from argparse import Namespace
from typing import List
from KAS import Path, Sampler
import cProfile

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from base import log, parser, dataset, models, trainer
from search.mcts import MCTSAlgorithm
from search.mcts.node import TreePath

def test_fast_update(path_repr, leaf_path_repr, model, sampler, args) -> None:
    path_serialized = TreePath.decode_str(path_repr).to_path()[0].serialize()
    leaf_path = TreePath.decode_str(leaf_path_repr)

    algo = MCTSAlgorithm(sampler, args)
    
    algo.load_eval_result(path_serialized, 0.5, leaf_path=leaf_path)


if __name__ == "__main__":
    log.setup(level=logging.DEBUG)
    
    args = parser.arg_parse()
    model, sampler = models.get_model(args, return_sampler=True)

    test_paths = [
        ("[Reduce(5560283196852294378), Merge(6736361731171789267), Merge(1456760626231759567)]", "[Reduce(5560283196852294378), Merge(6736361731171789267), Merge(1456760626231759567), Share(3105937193444993997), Share(3105937193444993970), Shift(7716808504002403245), Share(14368496488825318166), Expand(17580581500109511259), Expand(14858164344299864834), Shift(15939066619314697752), Share(10067379851896753133), Split(12602826350613247983), Shift(4729962409952852253), Finalize(13617644754852049647)]")
    ]

    for leaf_path, path in test_paths:
        cProfile.run("test_fast_update(path, leaf_path, model, sampler, args)")
