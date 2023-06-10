import torch

# Systems
import random

# KAS
from KAS.Bindings import CodeGenOptions


def parameters(args):

    use_cuda = torch.cuda.is_available()
    training_params = dict(
        val_period=1,
        use_cuda=use_cuda
    )
    sampler_params = dict(
        input_shape="[N,C_in,H,W]",
        output_shape="[N,C_out,H,W]",
        primary_specs=["N=64: 0", "H=32", "W=32"],
        coefficient_specs=["k_1=3: 4", "k_2=5: 4"],
        fixed_io_pairs=[(0, 0)],
        seed=random.SystemRandom().randint(
            0, 0x7fffffff) if args.kas_seed == 'pure' else args.seed,
        depth=args.kas_depth,
        dim_lower=args.kas_min_dim,
        dim_upper=args.kas_max_dim,
        maximum_tensors=2,
        maximum_reductions=4,
        max_flops=16420864000,  # manual conv size
        save_path=args.kas_sampler_save_dir,
        cuda=use_cuda,
        autoscheduler=CodeGenOptions.AutoScheduler.Anderson2021
    )

    extra_args = dict(
        max_macs=int(args.kas_max_macs * 1e9),
        min_macs=int(args.kas_min_macs * 1e9),
        max_model_size=int(args.kas_max_params * 1e6),
        min_model_size=int(args.kas_min_params * 1e6),
        prefix="",
        model_type="KASConv",  # TODO: dynamically load the module
        batch_size=args.batch_size,
        sample_input_shape=(args.batch_size, *args.input_size),
        device="cuda" if use_cuda else "cpu"
    )

    return training_params, sampler_params, extra_args
