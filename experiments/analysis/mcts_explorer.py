import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from base import log, models, parser, parser
from search import MCTSTree, MCTSExplorer, MCTSAlgorithm


if __name__ == "__main__":
    log.setup()

    # Arguments
    args = parser.arg_parse()

    # Sampler
    _, sampler = models.get_model(args, return_sampler=True)

    # Explorer
    if args.kas_mcts_explorer_path:
        with open(args.kas_mcts_explorer_path, "r") as file:
            mcts = MCTSTree.deserialize(
                json.load(file), sampler, keep_virtual_loss=True, keep_dead_state=False
            )
    else:
        mcts = MCTSAlgorithm(sampler, args).mcts
    explorer = MCTSExplorer(mcts)

    prefix = []
    if args.kas_mcts_explorer_script:
        assert os.path.exists(
            args.kas_mcts_explorer_script
        ), f"{args.kas_mcts_explorer_script} is not a file"
        with open(args.kas_mcts_explorer_script) as f:
            for command in f:
                if not command.startswith("#"):
                    prefix.append(command.rstrip("\n"))
    explorer.interactive(prefix=prefix)
