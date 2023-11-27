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
from KAS import Path, init_weights

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
            model = models.common.get_vanilla_common_model(args).cuda()
        else:
            assert hasattr(
                sys.modules[__name__], args.model
            ), f"Could not find model {args.model}"
            model_cls = getattr(sys.modules[__name__], args.model)
            model = model_cls().cuda()
        model.apply(init_weights)
        flops, params = thop.profile(
            model, (torch.ones((args.batch_size, *args.input_size), device="cuda"),)
        )
        flops /= args.batch_size
        if args.compile:
            model = torch.compile(model)
    else:
        try:
            model, sampler = models.get_model(args, return_sampler=True)
            node = sampler.visit(Path.deserialize(path))
            if node is None:
                logging.error(f"{Path.deserialize(path)} is not in the search space. ")
                for subpath in Path.deserialize(path).hierarchy:
                    if sampler.visit(subpath) is None:
                        logging.warning(f"Subpath {subpath} is not valid")
                        logging.info(f"Available Children of {node._node}:")
                        for child in node.get_children_handles():
                            child_node = node.get_child(child)
                            if child_node is None:
                                continue
                            logging.info(
                                f"\t{child}:\t{node.get_child_description(child)}"
                            )
                        break
                    else:
                        node = sampler.visit(subpath)
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

        accuracy = ImageNetTrainer.launch_from_args(
            model, args.imagenet_log_folder, args.batch_size
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

    save_dir = args.kas_server_save_dir

    node = sampler.visit(Path.deserialize(path)).to_node()

    os.makedirs(save_dir, exist_ok=True)

    acc_str = (
        ("0" * max(0, 5 - len(f"{int(accuracy * 10000)}"))) + f"{int(accuracy * 10000)}"
        if accuracy >= 0
        else "ERROR"
    )
    hash_str = f"{ctypes.c_size_t(hash(path)).value}"
    kernel_save_dir = os.path.join(save_dir, "_".join([acc_str, hash_str]))
    if not (os.path.exists(kernel_save_dir) and acc_str == "ERROR"):
        os.makedirs(kernel_save_dir, exist_ok=True)

        # GraphViz
        node.generate_graphviz_as_final(
            os.path.join(kernel_save_dir, "graph.dot"), "kernel"
        )

        # Loop
        with open(os.path.join(kernel_save_dir, "loop.txt"), "w") as f:
            f.write(str(node.get_nested_loops_as_final()))

        # Meta information
        with open(os.path.join(kernel_save_dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "path": path,
                    "accuracy": accuracy,
                    "flops": flops,
                    "params": nparams,
                    "kernel_flag": kernel_flag,
                    "loss": 0,
                },
                f,
                indent=2,
            )

        # copying kernel dir
        if kernel_flag != "MOCKPATH":
            shutil.copytree(
                sampler.realize(model, node).get_directory(),
                os.path.join(kernel_save_dir, "kernel_scheduler_dir"),
            )
