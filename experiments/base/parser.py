import argparse
import logging
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def arg_parse():
    parser = argparse.ArgumentParser(description="KAS trainer/searcher")

    # Resource
    parser.add_argument(
        "--server-mem-limit",
        type=float,
        default=1.0,
        help="Maximum portion of memory that server used. ",
    )
    parser.add_argument(
        "--client-mem-limit",
        type=float,
        default=1.0,
        help="Maximum portion of memory that client used. ",
    )

    # Evaluation
    parser.add_argument(
        "--evaluate-type",
        type=str,
        default="fp32",
        help="Type of re-evaluation (fp32 | ImageNet)",
    )
    parser.add_argument("--dirs", type=str, nargs="+", default=None)

    # Model
    parser.add_argument("--model", type=str, default="FCNet")
    parser.add_argument(
        "--kas-use-orig-model",
        default=False,
        action="store_true",
    )
    parser.add_argument("--num-classes", type=int, default=10)
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
    parser.add_argument(
        "--input-size",
        default=(3, 224, 224),
        nargs=3,
        type=int,
        metavar="N N N",
        help="Input all image dimensions (d h w, e.g. --input-size 3 224 224, "
        "model default if none)",
    )
    parser.add_argument(
        "--crop-pct",
        default=None,
        type=float,
        metavar="N",
        help="Input image center crop percent (for validation only)",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=IMAGENET_DEFAULT_MEAN,
        metavar="MEAN",
        help="Override mean pixel value of dataset",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=IMAGENET_DEFAULT_STD,
        metavar="STD",
        help="Override std deviation of of dataset",
    )
    parser.add_argument(
        "--interpolation",
        default="bilinear",
        type=str,
        metavar="NAME",
        help="Image resize interpolation type (overrides model)",
    )
    parser.add_argument(
        "--use-multi-epochs-loader",
        action="store_true",
        default=False,
        help="Use the multi-epochs-loader to save time at the beginning of every epoch",
    )

    # Dataset augmentations (used in Imagenet)
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="Disable all training augmentation, override other train aug args",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs="+",
        default=[0.08, 1.0],
        metavar="PCT",
        help="Random resize scale (default: 0.08 1.0)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs="+",
        default=[3.0 / 4.0, 4.0 / 3.0],
        metavar="RATIO",
        help="Random resize aspect ratio (default: 0.75 1.33)",
    )
    parser.add_argument(
        "--hflip",
        type=float,
        default=0.5,
        help="Horizontal flip training aug probability",
    )
    parser.add_argument(
        "--vflip",
        type=float,
        default=0.0,
        help="Vertical flip training aug probability",
    )
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)',
    )
    parser.add_argument(
        "--re-prob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--re-mode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--re-count", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--re-split",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="Mixup alpha, mixup enabled if > 0. (default: 0.8)",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="Cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="Cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bilinear",
        help='Training interpolation (random, bilinear, bicubic default: "random")',
    )
    parser.add_argument(
        "--tta",
        type=int,
        default=0,
        metavar="N",
        help="Test/inference time augmentation (oversampling) factor",
    )

    # Loss functions.
    parser.add_argument(
        "--jsd-loss",
        action="store_true",
        default=False,
        help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.",
    )
    parser.add_argument(
        "--bce-loss",
        action="store_true",
        default=False,
        help="Enable BCE loss w/ Mixup/CutMix use.",
    )
    parser.add_argument(
        "--bce-target-thresh",
        type=float,
        default=None,
        help="Threshold for binarizing softened BCE targets (default: None, disabled)",
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
    parser.add_argument("--grad-norm-clip", type=float, default=5.0)

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
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    # KAS preferences
    parser.add_argument("--kas-replace-placeholder", type=str, default=None)
    parser.add_argument("--kas-depth", default=12, type=int, help="KAS sampler depth")
    parser.add_argument(
        "--kas-max-enumerations",
        default=5,
        type=int,
        help="KAS sampler maximum enumerations per variable. ",
    )
    parser.add_argument(
        "--kas-max-variables-in-size",
        default=3,
        type=int,
        help="KAS sampler maximum different variables per size. ",
    )
    parser.add_argument(
        "--kas-max-size-multiplier",
        default=1,
        type=int,
        help="KAS sampler multiplier for maximum reduce size. ",
    )
    parser.add_argument(
        "--kas-max-weight-share-dim",
        default=8,
        type=int,
        help="Maximum size of the dim that is shared by 2 weights.",
    )
    parser.add_argument(
        "--kas-min-weight-share-dim",
        default=3,
        type=int,
        help="Minimum size of the dim that is shared by 2 weights.",
    )
    parser.add_argument(
        "--kas-max-chain-length",
        default=10,
        type=int,
        help="KAS sampler maximum primitive chain length. ",
    )
    parser.add_argument(
        "--kas-max-shift-rhs",
        default=5,
        type=int,
        help="KAS sampler maximum shift RHS size. ",
    )
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
        "--kas-max-expands",
        default=-1,
        type=int,
        help="KAS sampler maximum Splits",
    )
    parser.add_argument(
        "--kas-max-merges",
        default=-1,
        type=int,
        help="KAS sampler maximum Merges",
    )
    parser.add_argument(
        "--kas-max-splits",
        default=-1,
        type=int,
        help="KAS sampler maximum Splits",
    )
    parser.add_argument(
        "--kas-max-shifts",
        default=-1,
        type=int,
        help="KAS sampler maximum Shifts",
    )
    parser.add_argument(
        "--kas-max-strides",
        default=-1,
        type=int,
        help="KAS sampler maximum Strides",
    )
    parser.add_argument(
        "--kas-max-unfolds",
        default=-1,
        type=int,
        help="KAS sampler maximum Unfolds",
    )
    parser.add_argument(
        "--kas-max-shares",
        default=-1,
        type=int,
        help="KAS sampler maximum Shares",
    )
    parser.add_argument(
        "--kas-max-finalizations",
        default=5,
        type=int,
        help="KAS sampler maximum Finalizations",
    )
    parser.add_argument(
        "--kas-max-expansion-repeat-multiplier",
        default=1,
        type=int,
        help="KAS sampler maximum expansion multiplier",
    )
    parser.add_argument(
        "--kas-max-expansion-merge-multiplier",
        default=512,
        type=int,
        help="KAS sampler maximum expansion multiplier",
    )
    parser.add_argument(
        "--kas-no-exact-division",
        action="store_true",
        default=False,
        help="requires_exact_division=False",
    )
    parser.add_argument(
        "--kas-enable-even-unfold",
        action="store_true",
        default=False,
        help="requires_odd_kernel_size_in_unfold=False",
    )
    parser.add_argument(
        "--kas-min-unfold-ratio",
        default=1.5,
        type=float,
        help="KAS sampler minimum ratio between unfold kernel and unfolded dimension. ",
    )
    parser.add_argument(
        "--kas-max-pooling-factor",
        default=8,
        type=int,
        help="KAS sampler maximum pooling factor. ",
    )
    parser.add_argument(
        "--kas-scheduler-cache-dir",
        default=".scheduler-cache",
        help="KAS sampler saving directory",
    )
    parser.add_argument(
        "--kas-send-cache-dir",
        default=".send-cache",
        help="KAS send directory",
    )
    parser.add_argument(
        "--kas-client-cache-dir",
        default=".client-cache",
        help="KAS client cache directory",
    )
    parser.add_argument(
        "--kas-node-cache-dirs",
        default=".node-cache.txt",
        help="KAS evaluation result cache directory",
    )
    parser.add_argument(
        "--kas-stats-interval",
        default=600,
        type=int,
        help="KAS server statistic displaying interval (in seconds)",
    )
    parser.add_argument(
        "--kas-server-save-interval",
        default=600,
        type=int,
        help="KAS server saving interval (in seconds)",
    )
    parser.add_argument(
        "--kas-server-save-dir",
        default=".",
        type=str,
        help="KAS server saving directory",
    )
    parser.add_argument(
        "--kas-min-accuracy", default=0, type=float, help="Minimum accuracy for network"
    )
    parser.add_argument(
        "--kas-target", default="accuracy", type=str, help="Target metric of KAS"
    )
    parser.add_argument(
        "--kas-soft-flops-limit",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--kas-max-flops-ratio",
        default=0.5,
        type=float,
        help="Maximum FLOPs ratio to be counted as reward",
    )
    parser.add_argument(
        "--kas-min-flops-ratio",
        default=0.1,
        type=float,
        help="Maximum FLOPs ratio to be counted as reward",
    )
    parser.add_argument(
        "--kas-acc-lower-bound", default=0, type=float, help="Reward lower bound"
    )
    parser.add_argument(
        "--kas-acc-upper-bound", default=1, type=float, help="Reward upper bound"
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
        default=10000,
        type=float,
        help="Evaluation time limit (in seconds)",
    )
    parser.add_argument(
        "--kas-inference-time-limit",
        default=60,
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
        "--kas-num-virtual-evaluator",
        default=4,
        type=int,
        help="Number of virtual evaluators (for prefetching)",
    )
    parser.add_argument(
        "--kas-test-path",
        default="",
        type=str,
        help="Path to test",
    )
    parser.add_argument(
        "--kas-evaluate-lower-bound",
        default=0.75,
        type=float,
        help="Lower accuracy bound on kernels to re-evaluate. ",
    )

    # Pruning options
    parser.add_argument("--prune-milestones", default="", type=str)

    # GPT related
    parser.add_argument("--gpt-seq-len", default=None, type=int)
    parser.add_argument("--gpt-vocab-size", default=None, type=int)
    parser.add_argument("--gpt-tokenizer", default="gpt2-large", type=str)
    parser.add_argument("--gpt-max-iters", default=0, type=int)
    parser.add_argument("--gpt-max-minutes", default=0, type=float)
    parser.add_argument("--gpt-log-interval", default=10, type=int)
    parser.add_argument("--gpt-max-loss", default=3, type=float)
    parser.add_argument("--gpt-loss-output", default="", type=str)

    # Imagenet related
    parser.add_argument("--imagenet-log-folder", default="logs", type=str)
    parser.add_argument(
        "--imagenet-config-file",
        default="ffcv-imagenet/rn18_configs/rn18_88_epochs.yaml",
        type=str,
    )

    args = parser.parse_args()

    # Print
    args_str = "\n  > ".join([f"{k}: {v}" for k, v in vars(args).items()])
    logging.info(f"Execution arguments: \n  > {args_str}")

    if args.compile:
        import torch._dynamo.config

        torch._dynamo.config.suppress_errors = True

    return args
