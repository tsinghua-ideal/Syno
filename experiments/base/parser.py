import argparse
import logging
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .models import get_model_input_size


def arg_parse():
    parser = argparse.ArgumentParser(description="KAS trainer/searcher")

    # Model
    parser.add_argument("--model", type=str, default="FCNet")
    parser.add_argument(
        "--compile", action="store_true", default=False, help="Compile kernel"
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="torch/mnist")
    parser.add_argument("--root", type=str, default="~/data")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, metavar="N", help="Batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        metavar="N",
        help="How many training processes to use",
    )

    # Optimizer parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--opt",
        default="sgd",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "sgd"',
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Optimizer momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-3, help="Weight decay (default: 0.05)"
    )

    # Scheduler parameters
    parser.add_argument(
        "--sched", default="cosine", type=str, metavar="SCHEDULER", help="LR scheduler"
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="Warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="Lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="Number of epochs to train (default: 300)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        metavar="N",
        help="Epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=5,
        metavar="N",
        help="Epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--decay-rate", type=float, default=0.1, metavar="RATE", help="LR decay rate"
    )
    parser.add_argument(
        "--decay-milestones",
        type=int,
        default=[35, 65],
        nargs="+",
        metavar="RATE",
        help="LR decay milestones",
    )

    # KAS preferences
    parser.add_argument("--kas-replace-placeholder", type=str, default=None)
    parser.add_argument("--kas-depth", default=12, type=int, help="KAS sampler depth")
    parser.add_argument(
        "--kas-max-tensors", default=3, type=int, help="KAS sampler maximum tensors"
    )
    parser.add_argument(
        "--kas-max-reductions",
        default=4,
        type=int,
        help="KAS sampler maximum reductions",
    )
    parser.add_argument(
        "--kas-min-dim", default=1, type=int, help="KAS sampler minimum dimensions"
    )
    parser.add_argument(
        "--kas-max-dim", default=12, type=int, help="KAS sampler maximum dimensions"
    )
    parser.add_argument(
        "--kas-scheduler-cache-dir",
        default=".scheduler-cache",
        help="KAS sampler saving directory",
    )
    parser.add_argument(
        "--kas-server-save-interval",
        default=600,
        type=int,
        help="KAS server saving interval (in seconds)",
    )
    parser.add_argument(
        "--kas-server-save-dir",
        default=None,
        type=str,
        help="KAS server saving directory",
    )
    parser.add_argument(
        "--kas-max-flops",
        default=1e15,
        type=float,
        help="Maximum FLOPs for searched kernels (only for search)",
    )
    parser.add_argument(
        "--kas-min-accuracy", default=0, type=float, help="Minimum accuracy for network"
    )
    parser.add_argument(
        "--kas-target", default="accuracy", type=str, help="Target metric of KAS"
    )
    parser.add_argument(
        "--kas-flops-trunc",
        default=1e15,
        type=float,
        help="Maximum FLOPs to be counted as reward. ",
    )
    parser.add_argument(
        "--kas-reward-trunc", default=0, type=float, help="Reward lower bound"
    )
    parser.add_argument(
        "--kas-reward-power", default=2, type=float, help="Reward power"
    )

    # Search preferences
    parser.add_argument(
        "--kas-search-algo", default="MCTS", type=str, help="Search algorithm"
    )
    parser.add_argument(
        "--kas-server-addr", default="localhost", type=str, help="MCTS server address"
    )
    parser.add_argument(
        "--kas-server-port", default=8000, type=int, help="MCTS server port"
    )
    parser.add_argument("--kas-search-rounds", default=0, type=int, help="MCTS rounds")
    parser.add_argument(
        "--kas-mock-evaluate", action="store_true", default=False, help="Mock evaluate"
    )
    parser.add_argument(
        "--kas-resume",
        action="store_true",
        default=False,
        help="Resume previous training",
    )
    parser.add_argument(
        "--kas-retry-interval",
        default=10,
        type=float,
        help="Client retry time interval",
    )
    parser.add_argument(
        "--kas-mcts-explorer-path", default="", type=str, help="MCTS explorer path"
    )
    parser.add_argument(
        "--kas-mcts-explorer-script",
        default="",
        type=str,
        help="MCTS explorer script path",
    )
    parser.add_argument(
        "--kas-evaluate-time-limit",
        default=1800,
        type=float,
        help="Inference time limit (in seconds)",
    )
    parser.add_argument(
        "--kas-inference-time-limit",
        default=10,
        type=float,
        help="Inference time limit (in seconds)",
    )
    parser.add_argument(
        "--kas-sampler-workers",
        default=32,
        type=int,
        help="Number of workers for the sampler",
    )
    parser.add_argument(
        "--kas-mcts-workers",
        default=32,
        type=int,
        help="Number of workers for MCTS to simulate",
    )
    parser.add_argument(
        "--kas-num-virtual-evaluator",
        default=4,
        type=int,
        help="Number of virtual evaluators (for prefetching)",
    )

    args = parser.parse_args()

    # Extra arguments
    setattr(args, "input_size", get_model_input_size(args))

    # Print
    args_str = "\n  > ".join([f"{k}: {v}" for k, v in vars(args).items()])
    logging.info(f"Execution arguments: \n  > {args_str}")

    if args.compile:
        import torch._dynamo.config

        torch._dynamo.config.suppress_errors = True

    return args
