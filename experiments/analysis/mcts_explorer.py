import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from base import log, models, parser, parser
from search import MCTSTree, MCTSExplorer


if __name__ == '__main__':
    log.setup()

    # Arguments
    args = parser.arg_parse()

    # Sampler
    _, sampler = models.get_model(args, return_sampler=True)

    # Explorer
    with open(args.kas_mcts_explorer_path, 'r') as file:
        mcts = MCTSTree.deserialize(json.load(file), sampler)
    explorer = MCTSExplorer(mcts)
    
    explorer.interactive()
