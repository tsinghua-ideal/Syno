"""
Evaluate kernels in manual kernels. 
"""

import os, sys, json
import logging
from argparse import Namespace
from typing import List
from KAS import Path, Sampler
import thop, torch

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from base import (
    log,
    parser,
    dataset,
    models,
    trainer,
    device,
)


def train(
    args: Namespace,
    model: models.KASModel,
    sampler: Sampler,
    name: str,
    train_dataloader: dataset.FuncDataloader,
    val_dataloader: dataset.FuncDataloader,
    test_run: bool,
) -> None:
    if name == "Baseline":
        if args.model.startswith("torchvision/"):
            model = models.common.get_vanilla_common_model(args).cuda()
            flops, params = thop.profile(
                model, (torch.ones((args.batch_size, *args.input_size)))
            )
        else:
            assert hasattr(
                sys.modules[__name__], args.model
            ), f"Could not find model {args.model}"
            model_cls = getattr(sys.modules[__name__], args.model)
            model = model_cls().cuda()
            flops, params = model.profile(args.batch_size)
        logging.info(
            f"Loaded model has {flops / 1e9}G FLOPs per batch and {params / 1e6}M parameters in total."
        )
        message = "[Passed]"
        result = {
            "flops": flops,
            "params": params,
        }
    else:
        impl = models.ManualImpl(sampler)
        assert hasattr(impl, name), f"{name} is not a valid kernel"
        kernel = getattr(impl, name)()

        path = Path(kernel.convert_to_path(sampler))
        logging.info(f"Assembled path: {path}")
        if sampler.visit(path) is None:
            logging.warning(f"Path {path} is not valid, testing...")
            for subpath in path.hierarchy:
                if sampler.visit(subpath) is None:
                    logging.warning(f"Subpath {subpath} is not valid")
                    logging.info(f"Available Children of {node._node}:")
                    for child in node.get_children_handles():
                        child_node = node.get_child(child)
                        if child_node is None:
                            continue
                        logging.info(f"\t{child}:\t{node.get_child_description(child)}")
                    break
                else:
                    node = sampler.visit(subpath)
            Inspace = False
            message = "[Failed]"
        else:
            Inspace = True
            message = "[Passed]"

        result = {"path": str(path), "Inspace": Inspace}

        kernel_loader = sampler.realize(model, kernel, name)
        try:
            model.load_kernel(
                kernel_loader,
                compile=args.compile,
                batch_size=args.batch_size,
                seq_len=args.gpt_seq_len,
            )
            flops, params = model.profile(args.batch_size)
            logging.info(
                f"Loaded model has {flops / 1e9}G FLOPs per batch and {params / 1e6}M parameters in total."
            )
            result.update(
                {
                    "flops": flops,
                    "params": params,
                }
            )
        except Exception as e:
            logging.warning(e)

    if test_run:
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
            accuracy = max(trainer.train(model, train_dataloader, val_dataloader, args))
        print(f"Evaluation result: {flops} {params} {accuracy}")
        result["accuracy"] = accuracy

    print(f"{message} {name}")
    return result


def test_semantic_conv2d(test_kernels: List[str], test_run: bool) -> None:
    args = parser.arg_parse()
    device.initialize(args)

    logging.info("Loading dataset ...")
    train_dataloader, val_dataloader = dataset.get_dataloader(args)

    result_file = "base/unit_tests/results.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    if os.path.exists(result_file):
        with open(result_file) as f:
            results = json.load(f)
    else:
        results = {}

    model, sampler = models.get_model(args, return_sampler=True)

    for test_kernel in test_kernels:
        result = train(
            args,
            model,
            sampler,
            test_kernel,
            train_dataloader,
            val_dataloader,
            test_run,
        )
        if test_kernel not in results:
            results[test_kernel] = result
        else:
            results[test_kernel].update(result)

    with open("base/unit_tests/results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    log.setup(level=logging.INFO)

    test_kernels = [
        # "Baseline",
        "Conv2d_simple",
        # "Conv2d_dilation",
        "Conv2d_group",
        "Conv2d_group_oas",
        # "Conv2d_pool",
        # "Conv2d_pool1d",
        "Conv1d_shift1d",
        "Conv1d_transpose",
        "Conv1d_patch1d",
        "Shift2d",
        "kernel_07923",
    ]
    test_run = False

    test_semantic_conv2d(test_kernels, test_run)
