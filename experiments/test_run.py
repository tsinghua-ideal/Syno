import itertools
import logging
import random
import requests
import time
import sys
import torch
from KAS import Path

from base import log, models, parser, dataset, trainer


if __name__ == "__main__":
    log.setup(logging.INFO)

    args = parser.arg_parse()
    path = args.kas_test_path

    logging.info("Loading dataset ...")
    train_dataloader, val_dataloader = dataset.get_dataloader(args)

    logging.info("Preparing model ...")
    if args.kas_use_orig_model:
        if args.model.startswith("torchvision/"):
            model = models.common.get_common_model(args).cuda()
        else:
            assert hasattr(
                sys.modules[__name__], args.model
            ), f"Could not find model {args.model}"
            model_cls = getattr(sys.modules[__name__], args.model)
            model = model_cls().cuda()
        if args.compile:
            model = torch.compile(model)
    else:
        try:
            model, sampler = models.get_model(args, return_sampler=True)
            node = sampler.visit(Path.deserialize(path))
            if node is None:
                exit(1)
            node = node.to_node()
            if path:
                kernel_loader = sampler.realize(model, node)
                model.load_kernel(
                    kernel_loader,
                    compile=args.compile,
                    batch_size=args.batch_size,
                    seq_len=args.gpt_seq_len,
                )
        except Exception as e:
            if not "out of memory" in str(e):
                raise e
            logging.warning(f"OOM when evaluating {path}, skipping ...")
            model.remove_thop_hooks()
            flops, params, accuracy = 0, 0, -1
            exit(0)

    # Load and evaluate on a dataset
    flops, params = model.profile(args.batch_size)
    logging.info(
        f"Loaded model has {flops} FLOPs per batch and {params} parameters in total."
    )

    logging.info("Evaluating on real dataset ...")

    if "imagenet" in args.dataset:
        from fastargs import get_current_config
        from base.imagenet_trainer import ImageNetTrainer

        config = get_current_config()
        config.collect_config_file(args.imagenet_config_file)
        config.validate(mode="stderr")
        config.summary()

        accuracy = ImageNetTrainer.launch_from_args(model, args.imagenet_log_folder)
    else:
        accuracy = max(
            trainer.train(
                model,
                train_dataloader,
                val_dataloader,
                args,
            )
        )
    print(f"Evaluation result: {flops} {params} {accuracy}")
