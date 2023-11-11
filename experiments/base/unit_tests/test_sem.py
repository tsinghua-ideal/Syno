"""
Test for kernels' semantics. 
"""

import os, sys, shutil
import logging
import torch
from torch import nn
import random

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from base import log, parser, models

from KAS import Assembled, Sampler, KernelLoader


def Conv2d_simple(assembler) -> Assembled:
    N, H, W, k, C_in, C_out = assembler.get_sizes("N", "H", "W", "k_1", "C_in", "C_out")
    (
        in_N,
        in_H,
        in_W,
        in_C,
        out_C,
        w_in_C,
        w_k_1,
        w_k_2,
    ) = assembler.make_dims_of_sizes(N, H, W, C_in, C_out, C_in, k, k)

    main_H, windows_H = assembler.create_unfold(in_H, k)
    main_W, windows_W = assembler.create_unfold(in_W, k)

    shared_k_1 = assembler.create_share(windows_H, w_k_1)
    shared_k_2 = assembler.create_share(windows_W, w_k_2)
    shared_C_in = assembler.create_share(in_C, w_in_C)

    in_N.output(0)
    out_C.output(1)
    main_H.output(2)
    main_W.output(3)
    shared_k_1.sum()
    shared_k_2.sum()
    shared_C_in.sum()

    return assembler.assemble(
        "conv",
        "in_0 * in_1",
        [in_N, in_C, in_H, in_W],
        [out_C, w_in_C, w_k_1, w_k_2],
    )


def Conv2d_group_oas(assembler) -> Assembled:
    N, H, W, k_1, g, C_in, C_out = assembler.get_sizes(
        "N", "H", "W", "k_1", "s", "C_in", "C_out"
    )
    k = k_1
    (
        in_N,
        in_H,
        in_W,
        in_C,
        out_G,
        out_C_group,
        w_in_C,
        w_k_1,
        w_k_2,
    ) = assembler.make_dims_of_sizes(N, H, W, C_in, g, C_out / g, C_in / g, k, k)

    # Spatial dimensions
    main_H, windows_H = assembler.create_unfold(in_H, k)
    main_W, windows_W = assembler.create_unfold(in_W, k)

    shared_k_1 = assembler.create_share(windows_H, w_k_1)
    shared_k_2 = assembler.create_share(windows_W, w_k_2)

    # channel dimensions
    in_G, in_C_group = assembler.create_split(in_C, C_in / g)

    shared_G = assembler.create_share(in_G, out_G)
    shared_C_in = assembler.create_share(in_C_group, w_in_C)

    tmp_dim = assembler.create_expand(C_out / g)
    out_C_group_masked = assembler.create_share(tmp_dim, out_C_group)
    final_C_out = assembler.create_merge(shared_G, out_C_group_masked)

    in_N.output(0)
    final_C_out.output(1)
    main_H.output(2)
    main_W.output(3)
    shared_k_1.sum()
    shared_k_2.sum()
    shared_C_in.sum()

    return assembler.assemble(
        "conv",
        "in_0 * in_1",
        [in_N, in_C, in_H, in_W, tmp_dim],
        [out_G, out_C_group, w_in_C, w_k_1, w_k_2],
    )


def test_semantics(semantic_map) -> None:
    args = parser.arg_parse()

    model = models.get_common_model(args).cuda()
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
        "maximum_enumerations_per_var": args.kas_max_enumerations,
        "max_rdom_size_multiplier": args.kas_max_size_multiplier,
        "save_path": args.kas_scheduler_cache_dir,
        "cuda": True,
        "num_worker_threads": args.kas_sampler_workers,
        "requires_exact_division": False,
        "requires_odd_kernel_size_in_unfold": False,
        "minimum_unfold_ratio": 1.0,
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
    sampler = Sampler(**params)

    mappings = [{"N": 2, "C_in": 64, "C_out": 128, "H": 8, "W": 8}]

    placeholders = [
        models.placeholder.ConvPlaceholder(mapping["C_in"], mapping["C_out"], 3)
        for mapping in mappings
    ]
    for pl in placeholders:
        pl.cuda()

    for test_kernel_func, weight_func, referred_layer_func in semantic_map:
        kernel = test_kernel_func(sampler.create_assembler())
        kernel_packs = KernelLoader(
            sampler._realize(kernel, mappings, "test_semantics")
        ).construct_kernel_packs()

        for mapping, placeholder, kernel_pack in zip(
            mappings, placeholders, kernel_packs
        ):
            placeholder.reload(kernel_pack, args.compile)
            layer = referred_layer_func(mapping)
            print("Weights0:", kernel_pack.weights[0])
            print("Weights1:", layer.weight)
            layer.weight = weight_func(kernel_pack.weights[0])
            layer = layer.cuda()
            input = torch.rand(
                (mapping["N"], mapping["C_in"], mapping["H"], mapping["W"]),
                device="cuda",
            )
            output_p = placeholder(input)
            output_l = layer(input)
            assert (
                output_p.size() == output_l.size()
            ), f"Size mismatch: {output_p.size()} != {output_l.size()}"
            assert torch.all(
                torch.abs(output_p - output_l) <= 0.03
            ), f"Value mismatch: {output_p} != {output_l}, {torch.max(torch.abs(output_p - output_l))}"


if __name__ == "__main__":
    log.setup(level=logging.INFO)
    shutil.rmtree(".test-scheduler-cache")

    semantic_map = [
        (
            Conv2d_group_oas,
            lambda weight: nn.Parameter(
                torch.flatten(
                    torch.permute(
                        weight,
                        (
                            0,
                            4,
                            1,
                            2,
                            3,
                        ),
                    ),
                    0,
                    1,
                )
            ),
            lambda mapping: nn.Conv2d(
                mapping["C_in"],
                mapping["C_out"],
                kernel_size=3,
                padding=1,
                groups=32,
                bias=False,
            ),
        ),
        (
            Conv2d_simple,
            lambda weight: nn.Parameter(weight),
            lambda mapping: nn.Conv2d(
                mapping["C_in"], mapping["C_out"], kernel_size=3, padding=1, bias=False
            ),
        ),
    ]

    test_semantics(semantic_map)
