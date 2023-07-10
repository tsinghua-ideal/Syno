import logging
from utils import models, parser, explorer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Parsing arguments ......")
    args = parser.arg_parse()
    print("Acquiring models ......")
    model, sampler = models.get_model(args, return_sampler=True)
    print("Setup explorer ......")
    explorer.explore(sampler, "analysis/mcts.json")
