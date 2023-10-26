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

    algo = MCTSAlgorithm(sampler, model, args)
    
    algo.load_eval_result(path_serialized, 0.5, leaf_path=leaf_path)


if __name__ == "__main__":
    log.setup(level=logging.DEBUG)
    
    args = parser.arg_parse()
    model, sampler = models.get_model(args, return_sampler=True)

    test_paths = [
        ("[Reduce(3815673689317161159), Reduce(16944764544590353949)]", "[Reduce(3815673689317161159), Reduce(16944764544590353949), Merge(17926165022425061937), Merge(1301728425438608103), Split(4242077998742030658), Share(15302844813840757071), Shift(7716808504002403245), Split(3190732959757924985), Share(13497865469262156188), Expand(629380456128670887), Shift(4729962409952852253), Share(14765397666072454195), Share(5592658499789034655), Expand(9911227917462026532), Finalize(4723397394311951784)]")
    ]

    for leaf_path, path in test_paths:
        cProfile.run("test_fast_update(path, leaf_path, model, sampler, args)")
