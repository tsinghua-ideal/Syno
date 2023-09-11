import logging
import random
import sys
import torch
from typing import Tuple, Optional, Union
from KAS import Sampler, Path
from KAS.Bindings import CodeGenOptions
from KAS.Placeholder import build_placeholder_mappings, remove_unsatisfied_placeholders

from . import placeholder
from .model import KASModel
from .conv_net import ConvNet, SpeedyResNet
from .fc_net import FCNet
from .common import get_common_model


def get_sampler(args, model) -> Sampler:
    # FLOPs ratio
    raw_flops, _ = model.profile(not_count_placeholder=True)
    flops, _ = model.profile()
    max_flops = args.kas_max_flops_ratio * flops
    assert max_flops > raw_flops, f"Maximum FLOPs {max_flops} is smaller than raw FLOPs {raw_flops}"
    max_placeholder_flops = max_flops - raw_flops
    logging.info(f"Raw FLOPs per batch: {raw_flops / 1e9:.5f}G")
    logging.info(f"Maximum total placeholder FLOPs per batch: {max_placeholder_flops / 1e9:.5f}G")
    logging.info(f"Enable soft FLOPs limit: {args.kas_soft_flops_limit}")
    setattr(args, "original_flops", flops)

    if args.kas_soft_flops_limit:
        # TODO: maybe change to zero
        max_placeholder_flops = 1e12

    # Build sampler09
    model_params = model.sampler_parameters()
    params = {
        "input_shape": model_params["input_shape"],
        "output_shape": model_params["output_shape"],
        "primary_specs": model_params["primary_specs"],
        "coefficient_specs": model_params["coefficient_specs"],
        "fixed_io_pairs": model_params["fixed_io_pairs"],
        "seed": random.SystemRandom().randint(0, 0x7FFFFFFF)
        if args.seed is None
        else args.seed,
        "depth": args.kas_depth,
        "dim_lower": args.kas_min_dim,
        "dim_upper": args.kas_max_dim,
        "maximum_tensors": args.kas_max_tensors,
        "maximum_reductions": args.kas_max_reductions,
        "maximum_shifts": args.kas_max_shifts,
        "max_flops": max_placeholder_flops * args.batch_size,
        "save_path": args.kas_scheduler_cache_dir,
        "cuda": True,
        "autoscheduler": CodeGenOptions.AutoScheduler.Anderson2021,
        "num_worker_threads": args.kas_sampler_workers,
        "requires_exact_division": True,
        "requires_odd_kernel_size_in_unfold": True,
        "minimum_unfold_ratio": 1.5,
        "extra_options": {
            "beam_size": "32",
            "num_passes": "1",
            "parallelism": "82",
            "shared_memory_limit_kb": "48",
            "shared_memory_sm_limit_kb": "100",
            "active_block_limit": "512",
            "active_warp_limit": "1024",
            "search_space_options": "1000",
        },
    }
    sampler = Sampler(net=model, **params)
    sampler._bind_debug_context()
    return sampler


def get_model(
    args, return_sampler=False
) -> Union[Tuple[KASModel, Optional[Sampler]], KASModel]:
    # Create model instance
    if args.model.startswith("torchvision/"):
        model = get_common_model(args).cuda()
    else:
        assert hasattr(sys.modules[__name__], args.model), f"Could not find model {args.model}"
        model_cls = getattr(sys.modules[__name__], args.model)
        model = model_cls().cuda()
    flops, params = model.profile()
    logging.info(
        f"Base model {args.model} has {flops / 1e9:.5f}GFLOPs (per batch) and {params / 1e6:.2f}M parameters"
    )

    # Build mapping for usages
    sample_input = torch.randn((args.batch_size, *model.sample_input_shape())).cuda()
    build_placeholder_mappings(model, sample_input)
    count = remove_unsatisfied_placeholders(model)
    logging.info(f"Recovered {count} unsatisfied placeholders")

    # Build sampler
    logging.info("Building sampler ...")
    sampler = (
        get_sampler(args, model)
        if (args.kas_replace_placeholder or return_sampler)
        else None
    )

    # Replace kernel
    if args.kas_replace_placeholder is not None:
        logging.info(f"Replacing kernel with {args.kas_replace_placeholder} ...")
        cls_name = args.kas_replace_placeholder.capitalize() + "Placeholder"
        assembled = getattr(placeholder, cls_name).impl(sampler.create_assembler())
        logging.debug(f"Assembled path: {assembled.convert_to_path(sampler)}")
        logging.debug(f"Assembled path (serialized): {Path(assembled.convert_to_path(sampler)).serialize()}")
        if sampler.visit(assembled.convert_to_path(sampler)) is None:
            path = Path(assembled.convert_to_path(sampler))
            logging.warn(f"Path {path} is not valid, testing...")
            for subpath in path.hierarchy:
                if sampler.visit(subpath) is None:
                    logging.warn(f"Subpath {subpath} is not valid")
                    break
        model.load_kernel(
            sampler,
            assembled,
            args.kas_replace_placeholder,
            compile=args.compile,
            batch_size=args.batch_size,
        )
        flops_replaced, params_replaced = model.profile(batch_size=args.batch_size, force_update=True)
        flops_base, params_base = model.profile(batch_size=args.batch_size, force_update=True, not_count_placeholder=True)
        logging.info(f"Replaced model {args.model} has {flops_replaced / 1e9:.5f}G FLOPs and {params_replaced / 1e6:.2f}M parameters")
        logging.info(f"Placeholder flops change {flops - flops_base:.2f} -> {flops_replaced - flops_base:.2f}")
        logging.info(f"Placeholder params change {params - params_base:.2f} -> {params_replaced - params_base:.2f}")

    if return_sampler:
        assert sampler
    return (model, sampler) if return_sampler else model
