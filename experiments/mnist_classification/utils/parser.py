import argparse
import time


def arg_parse():
    parser = argparse.ArgumentParser(
        description='KAS MNIST trainer/searcher')

    # Dataset.
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='Random seed (default: 42)')
    parser.add_argument('--batch-size', metavar='N', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--num-classes', metavar='N', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--input-size', default=(1, 28, 28), nargs=3, type=int, metavar='N N N',
                        help='Input all image dimensions (d h w, e.g. --input-size 3 224 224, '
                             'model default if none)')
    parser.add_argument('--mean', type=float, nargs='+', default=0.1307, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=0.3081, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('-j', '--num-workers', type=int, default=2, metavar='N',
                        help='How many training processes to use (default: 2)')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                        help='Use the multi-epochs-loader to save time at the beginning of every epoch')

    # Dataset augmentation. (Unused)
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--re-prob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--re-mode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--re-count', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--re-split', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='Mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='Cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='Cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--tta', type=int, default=0, metavar='N',
                        help='Test/inference time augmentation (oversampling) factor')

    # Optimizer parameters.
    parser.add_argument('--lr', type=float, default=5., metavar='LR',
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: none, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode, one of ("norm", "value", "agc")')

    # Scheduler parameters.
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='PCT, PCT',
                        help='Learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='Learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='Learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='Learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='Amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='Learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='Learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.01, metavar='LR',
                        help='Warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=45, metavar='N',
                        help='Number of epochs to train (default: 300)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='Epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                        help='Epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='Patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--decay-milestones', '--dm', default=[
                        25, 45], nargs='+', metavar='RATE', help='LR decay milestones (default: 100, 150)')

    # Misc.
    parser.add_argument('--forbid-eval-nan', action='store_true',
                        help='Whether to forbid NaN during evaluation')
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='Path to output folder (default: none, current dir, training only)')
    parser.add_argument('--resume', metavar='PATH', type=str, default='',
                        help='Path to the checkpoint for resuming (only for training)')
    parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                        help='Number of checkpoints to keep (default: 10)')
    parser.add_argument('--native-amp', action='store_true',
                        help='Whether to use PyTorch native AMP')
    parser.add_argument('--apex-amp', action='store_true',
                        help='Whether to use NVIDIA Apex AMP')
    parser.add_argument('--apex-amp-loss-scale', default=0.0, type=float,
                        help='Loss scale for Apex AMP (0 for dynamic scaling)')
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1"')
    parser.add_argument('--log-interval', default=100, type=int, metavar='INTERVAL',
                        help='Logging interval')

    # Distributed.
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off')

    # KAS preferences.
    parser.add_argument('--kas-load-checkpoint', default='', type=str,
                        help='Path to checkpoint file (after replacing the kernel)')
    parser.add_argument('--kas-weight-sharing', action='store_true',
                        help='Enable weight sharing with the best model in the search')
    parser.add_argument('--kas-rounds', default=0, type=int,
                        help='Search rounds for KAS (only for search)')
    parser.add_argument('--kas-seed', default='pure', type=str,
                        help='KAS seed settings (one of "global" and "pure"), '
                             '"global" means same with training, "pure" means purely random '
                             '(only for search)')
    parser.add_argument('--kas-first-epoch-pruning-milestone', default='', type=str,
                        help='First epoch milestone pruning')
    parser.add_argument('--kas-epoch-pruning-milestone', default='', type=str,
                        help='Epoch accuracy milestone pruning')
    parser.add_argument('--kas-log-dir', default='', type=str,
                        help='KAS logging directory')
    parser.add_argument('--kas-oss-bucket', default='', type=str,
                        help='Log into OSS buckets')
    parser.add_argument('--kas-bmm-pct', default=0.1, type=float,
                        help='Possibility to forcibly contain BMM (attention-like, only for search)')
    parser.add_argument('--kas-proxy-root', default='', metavar='DIR', type=str,
                        help='Path to proxy dataset (only for search)')
    parser.add_argument('--kas-proxy-threshold', default=70.0, type=float,
                        help='Proxy dataset threshold for real training (only for search)')
    parser.add_argument('--kas-kernel', default='', type=str,
                        help='Path to the replaced kernel (only for training)')
    parser.add_argument('--kas-depth', default=4,
                        type=int, help='kas sampler depth')
    parser.add_argument('--kas-min-dim', default=1,
                        type=int, help='kas sampler minimum dimensions')
    parser.add_argument('--kas-max-dim', default=8,
                        type=int, help='kas sampler maximum dimensions')
    parser.add_argument('--kas-sampler-save-dir', default='./samples',
                        help='Sampler saving directory')
    parser.add_argument('--kas-searcher-type', default='mcts', type=str,
                        help='searcher type (mcts or random)')
    parser.add_argument('--result-save-dir', default='./results',
                        help='Sampler saving directory')
    parser.add_argument('--kas-iterations', default=30, type=int,
                        help='Searcher iterations')
    parser.add_argument('--kas-simulate-retry-limit', default=10000, type=int,
                        help='simulate max round')
    parser.add_argument('--kas-leaf-parallelization-number', default=1, type=int,
                        help='leaf parallelization')
    # https://github.com/CyCTW/Parallel-MCTS/blob/master/src/MCTS.h
    parser.add_argument('--kas-tree-parallelization-virtual-loss-constant', default=1.0, type=float,
                        help='virtual-loss-constant of tree parallelization')
    parser.add_argument('--kas-min-macs', default=0, type=float,
                        help='Minimum MACs for searched kernels (in G-unit, only for search)')
    parser.add_argument('--kas-max-macs', default=0.8, type=float,
                        help='Maximum MACs for searched kernels (in G-unit, only for search)')
    parser.add_argument('--kas-min-params', default=0, type=float,
                        help='Minimum params for searched kernels (in M-unit, only for search)')
    parser.add_argument('--kas-max-params', default=3, type=float,
                        help='Maximum params for searched kernels (in M-unit, only for search)')
    parser.add_argument('--kas-min-receptive-size', default=1, type=int,
                        help='Minimum receptive size (only for search)')
    parser.add_argument('--kas-min-proxy-kernel-scale', default=0.2, type=float,
                        help='Minimum kernel scale (geometric mean, only for search)')
    parser.add_argument('--kas-sampling-workers', default=10, type=int,
                        help='Workers to use for sampling (only for search)')
    parser.add_argument('--kas-proxy-kernel-scale-limit', default=0.3, type=float,
                        help='Minimum/maximum kernel scale (geometric mean, only for search)')
    parser.add_argument('--kas-selector-address', default='http://127.0.0.1:8000', type=str,
                        help='Selector server address')
    parser.add_argument('--kas-selector-max-params',
                        default=6, help='Maximum model size')
    parser.add_argument('--kas-selector-dir', default='',
                        help='Selector working directory')
    parser.add_argument('--kas-selector-save-dir', default='./kas_save',
                        help='Selector saving directory')

    parser.add_argument('--host', type=str, metavar='HOST', default='0.0.0.0')
    parser.add_argument('--port', type=int, metavar='PORT', default='8000')

    # Parse program arguments, add timestamp information, and checks.
    args = parser.parse_args()
    setattr(args, 'timestamp', time.time_ns())
    assert args.kas_min_macs <= args.kas_max_macs, 'Minimum FLOPs should be lower than maximum'
    assert args.kas_min_params <= args.kas_max_params, 'Minimum params should be lower than maximum'
    assert args.kas_min_dim <= args.kas_max_dim, 'Minimum params should be lower than maximum'
    assert not (
        args.apex_amp and args.native_amp), 'Can not enable both native/Apex AMP'
    if args.apex_amp_loss_scale == 0:
        args.apex_amp_loss_scale = None
    return args
