import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from base import log, models, parser, parser
from KAS import Explorer

if __name__ == '__main__':
    log.setup()

    # Arguments
    args = parser.arg_parse()

    # Sampler
    _, sampler = models.get_model(args, return_sampler=True)

    # Explorer
    explorer = Explorer(sampler)

    explorer.interactive()
