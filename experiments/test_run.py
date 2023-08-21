import itertools
import logging
import random
import requests
import time
from KAS import Path

from base import log, models, parser, dataset, trainer


if __name__ == "__main__":
    log.setup()

    args = parser.arg_parse()
    path = args.kas_test_path

    logging.info("Preparing model ...")
    model, sampler = models.get_model(args, return_sampler=True)

    logging.info("Loading dataset ...")
    train_dataloader, val_dataloader = dataset.get_dataloader(args)

    try:
        node = sampler.visit(Path.deserialize(path)).to_node()

        # Load and evaluate on a dataset
        try:
            model.load_kernel(
                sampler, node, compile=args.compile, batch_size=args.batch_size
            )
            flops, params = model.profile(args.batch_size)
            logging.debug(
                f"Loaded model has {flops} FLOPs per batch and {params} parameters in total."
            )

            logging.info("Evaluating on real dataset ...")
            accuracy = max(trainer.train(model, train_dataloader, val_dataloader, args))
        except Exception as e:
            if not "out of memory" in str(e):
                raise e
            logging.warning(f"OOM when evaluating {path}, skipping ...")
            model.remove_thop_hooks()
            flops, params, accuracy = 0, 0, -1
        print(f"Evaluation result: {flops} {params} {accuracy}")
    except KeyboardInterrupt:
        logging.info("Interrupted by user, exiting ...")
