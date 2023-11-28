import itertools
import logging
import random
import requests
import time
import os, sys, json
import ctypes
import shutil
import torch
import thop
from KAS import Path, init_weights, KernelLoader

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
            model = models.common.get_vanilla_common_model(args)
        else:
            assert hasattr(
                sys.modules[__name__], args.model
            ), f"Could not find model {args.model}"
            model_cls = getattr(sys.modules[__name__], args.model)
            model = model_cls()
        model.apply(init_weights)
        flops, params = thop.profile(
            model, (torch.ones((args.batch_size, *args.input_size)),)
        )
        flops /= args.batch_size
        if args.compile:
            torch._dynamo.reset()
            model = torch.compile(model)
    else:
        try:
            kernel_loader = KernelLoader.from_directory(os.path.join(path, "kernel_scheduler_dir"))
            model = models.get_model(args, return_sampler=False)
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

        accuracy = ImageNetTrainer.launch_from_args(
            model, args.imagenet_log_folder, args.batch_size, args.imagenet_config_file
        )
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

    nparams = params
    kernel_flag = "LOAD_SUCCESS"

    # Meta information
    with open(os.path.join(path, "test_run_meta.json"), "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "flops": flops,
                "params": nparams,
                "kernel_flag": kernel_flag,
                "loss": 0,
            },
            f,
            indent=2,
        )
