import logging
import random
import os, sys
import torch
from typing import Tuple, Optional, Union
from KAS import Sampler, Path, CodeGenOptions, KernelLoader
from KAS.Placeholder import build_placeholder_mappings, remove_unsatisfied_placeholders

from . import placeholder
from .model import KASModel
from .conv_net import ConvNet, SpeedyResNet
from .fc_net import FCNet
from .common import get_common_model, get_vanilla_common_model
from .gpt import GPTConfig, GPT
from .gcn import GCN
from .mobilenetv2 import MobileNetV2
from .manual_kernels import ManualImpl


def get_sampler(args, model) -> Sampler:
    # FLOPs ratio
    raw_flops, _ = model.profile(not_count_placeholder=True, seq_len=args.gpt_seq_len)
    flops, _ = model.profile(seq_len=args.gpt_seq_len)
    min_flops = args.kas_min_flops_ratio * flops
    max_flops = args.kas_max_flops_ratio * flops
    max_flops = 1e12 if max_flops == 0 else max_flops
    assert (
        max_flops > raw_flops
    ), f"Maximum FLOPs {max_flops} is smaller than raw FLOPs {raw_flops}"
    if min_flops < raw_flops:
        min_flops = raw_flops
    max_placeholder_flops = max_flops - raw_flops
    min_placeholder_flops = min_flops - raw_flops
    logging.info(f"Raw FLOPs per batch: {raw_flops / 1e9:.5f}G")
    logging.info(
        f"Maximum total placeholder FLOPs per batch: {max_placeholder_flops / 1e9:.5f}G"
    )
    logging.info(
        f"Minimum total placeholder FLOPs per batch: {min_placeholder_flops / 1e9:.5f}G"
    )
    logging.info(f"Enable soft FLOPs limit: {args.kas_soft_flops_limit}")
    setattr(args, "original_flops", flops)

    if args.kas_soft_flops_limit:
        # TODO: maybe change to zero
        max_placeholder_flops = 1e12

    # Build sampler
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
        "maximum_tensors": args.kas_max_tensors,
        "maximum_reductions": args.kas_max_reductions,
        "maximum_expands": args.kas_max_expands,
        "maximum_merges": args.kas_max_merges,
        "maximum_splits": args.kas_max_splits,
        "maximum_shifts": args.kas_max_shifts,
        "maximum_strides": args.kas_max_strides,
        "maximum_unfolds": args.kas_max_unfolds,
        "maximum_shares": args.kas_max_shares,
        "maximum_finalizations": args.kas_max_finalizations,
        "max_expansion_repeat_multiplier": args.kas_max_expansion_repeat_multiplier,
        "max_expansion_merge_multiplier": args.kas_max_expansion_merge_multiplier,
        "max_expansion_weights_sharing_dim_size": args.kas_max_weight_share_dim,
        "min_expansion_weights_sharing_dim_size": args.kas_min_weight_share_dim,
        "maximum_valid_reshape_shift_pattern": args.kas_max_shift_rhs,
        "max_flops": max_placeholder_flops * args.batch_size,
        "min_flops": min_placeholder_flops * args.batch_size,
        "max_vram": int(args.client_mem_limit * 40 * 1024),
        "maximum_enumerations_per_var": args.kas_max_enumerations,
        "maximum_variables_in_size": args.kas_max_variables_in_size,
        "max_chain_length": args.kas_max_chain_length,
        "max_rdom_size_multiplier": args.kas_max_size_multiplier,
        "min_single_weight_params": args.kas_min_weight_nparams,
        "disallow_tile": not args.kas_allow_tile,
        "save_path": os.path.join(
            args.kas_server_save_dir, args.kas_scheduler_cache_dir
        ),
        "cuda": True,
        "autoscheduler": CodeGenOptions.AutoScheduler.Anderson2021,
        "num_worker_threads": args.kas_sampler_workers,
        "requires_exact_division": not args.kas_no_exact_division,
        "requires_odd_kernel_size_in_unfold": not args.kas_enable_even_unfold,
        "minimum_unfold_ratio": args.kas_min_unfold_ratio,
        "max_pooling_factor": args.kas_max_pooling_factor,
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
    logging.info(f"Sampler parameters={params}")
    sampler = Sampler(net=model, **params)
    sampler._bind_debug_context()
    return sampler


def get_vanilla_model(args) -> torch.nn.Module:
    if args.model.startswith("torchvision/"):
        return get_vanilla_common_model(args)
    else:
        assert False, f"Could not find model {args.model}"


def get_model(
    args, return_sampler=False, sample_input=None
) -> Union[Tuple[KASModel, Optional[Sampler]], KASModel]:
    # Create model instance
    if args.model.startswith("torchvision/"):
        model = get_common_model(args)
    elif args.model.startswith("gpt/"):
        from transformers import GPT2Tokenizer
        config = GPT.get_default_config()
        config.model_type = args.model[len("gpt/") :]
        config.vocab_size = GPT2Tokenizer.from_pretrained(args.gpt_tokenizer).vocab_size
        config.block_size = args.gpt_seq_len
        model = GPT(config)
    elif args.model == "gcn":
        model = GCN()
    elif args.model == "mobilenet_v2":
        model = MobileNetV2(args.num_classes, args.input_size)
    else:
        assert hasattr(
            sys.modules[__name__], args.model
        ), f"Could not find model {args.model}"
        model_cls = getattr(sys.modules[__name__], args.model)
        model = model_cls()
    flops, params = model.profile(seq_len=args.gpt_seq_len)
    logging.info(
        f"Base model {args.model} has {flops / 1e9:.5f} GFLOPs (per batch) and {params / 1e6:.2f}M parameters"
    )

    # Build mapping for usages
    if sample_input is None:
        sample_input = torch.ones(
            (args.batch_size, *model.sample_input_shape(args.gpt_seq_len))
        )
        if args.gpt_seq_len:
            sample_input = sample_input.long()
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
    logging.info(f"Mappings: {Sampler._extract_all_mappings(model)}")
    if sampler:
        if any(
            isinstance(pl.referred_layer, torch.nn.Conv2d)
            for pl in sampler._extract_placeholders(model)
        ):
            logging.info(
                f"Kernel sizes: {[pl.referred_layer.kernel_size for pl in sampler._extract_placeholders(model)]}"
            )
            logging.info(
                f"Groups: {[pl.referred_layer.groups for pl in sampler._extract_placeholders(model)]}"
            )
            logging.info(
                f"Strides: {[pl.referred_layer.stride for pl in sampler._extract_placeholders(model)]}"
            )

    # Replace kernel
    if args.kas_replace_placeholder is not None:
        logging.info(f"Replacing kernel with {args.kas_replace_placeholder} ...")
        if os.path.isdir(args.kas_replace_placeholder):
            logging.info(f"Loading from directory ...")
            kernel_directory = os.path.join(
                args.kas_replace_placeholder, "kernel_scheduler_dir"
            )
            kernel_loader = KernelLoader.from_directory(kernel_directory)
        else:
            logging.info(f"Loading from class ...")
            cls_name = args.kas_replace_placeholder + "Placeholder"
            assembler = sampler.create_assembler()
            assembled = getattr(placeholder, cls_name).impl(assembler)
            logging.debug(f"Assembled path: {assembled.convert_to_path(sampler)}")
            logging.debug(
                f"Assembled path (serialized): {Path(assembled.convert_to_path(sampler)).serialize()}"
            )
            if sampler.visit(assembled.convert_to_path(sampler)) is None:
                path = Path(assembled.convert_to_path(sampler))
                logging.warning(f"Path {path} is not valid, testing...")
                for subpath in path.hierarchy:
                    if sampler.visit(subpath) is None:
                        logging.warning(f"Subpath {subpath} is not valid")
                        break
            kernel_loader = sampler.realize(
                model, assembled, args.kas_replace_placeholder
            )
        model.load_kernel(
            kernel_loader,
            compile=args.compile,
            batch_size=args.batch_size,
            seq_len=args.gpt_seq_len,
        )
        flops_replaced, params_replaced = model.profile(
            batch_size=args.batch_size, force_update=True, seq_len=args.gpt_seq_len
        )
        flops_base, params_base = model.profile(
            batch_size=args.batch_size,
            force_update=True,
            not_count_placeholder=True,
            seq_len=args.gpt_seq_len,
        )
        logging.info(
            f"Replaced model {args.model} has {flops_replaced / 1e9:.5f}G FLOPs and {params_replaced / 1e6:.2f}M parameters"
        )
        logging.info(
            f"Placeholder flops change {flops - flops_base:.2f} -> {flops_replaced - flops_base:.2f}"
        )
        logging.info(
            f"Placeholder params change {params - params_base:.2f} -> {params_replaced - params_base:.2f}"
        )

    if return_sampler:
        assert sampler
    return (model, sampler) if return_sampler else model
