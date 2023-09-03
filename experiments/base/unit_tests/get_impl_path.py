import os, sys
sys.path.append(os.getcwd())

from base import log, parser, models

if __name__ == "__main__":
    log.setup()
    args = parser.arg_parse()
    model = models.get_model(args, return_sampler=False)