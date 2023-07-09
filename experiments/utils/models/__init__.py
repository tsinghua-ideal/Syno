import logging
import random
import sys
import torch
from thop import profile
from KAS import Sampler
from KAS.Bindings import CodeGenOptions
from KAS.Placeholder import build_placeholder_mappings

from . import placeholder
from .model import KASModel
from .conv_net import ConvNet
from .fc_net import FCNet


def get_model_input_size(args):
    assert hasattr(sys.modules[__name__], args.model), f'Could not find model {args.model}'
    model_cls = getattr(sys.modules[__name__], args.model)
    return model_cls.sample_input_shape()


def get_sampler(args, model):
    # Build sampler
    model_params = model.sampler_parameters()
    params = {
        'input_shape': model_params['input_shape'],
        'output_shape': model_params['output_shape'],
        'primary_specs': model_params['primary_specs'],
        'coefficient_specs': model_params['coefficient_specs'],
        'fixed_io_pairs': model_params['fixed_io_pairs'],
        'seed': random.SystemRandom().randint(0, 0x7fffffff) if args.seed is None else args.seed,
        'depth': args.kas_depth,
        'dim_lower': args.kas_min_dim,
        'dim_upper': args.kas_max_dim,
        'maximum_tensors': args.kas_max_tensors,
        'maximum_reductions': args.kas_max_reductions,
        'max_flops': args.kas_max_flops,
        'save_path': args.kas_sampler_save_dir,
        'cuda': True,
        'autoscheduler': CodeGenOptions.AutoScheduler.Anderson2021,
        'extra_options': {
            'beam_size': '32',
            'num_passes': '1',
            'parallelism': '82',
            'shared_memory_limit_kb': '48',
            'shared_memory_sm_limit_kb': '100',
            'active_block_limit': '512',
            'active_warp_limit': '1024',
            'search_space_options': '1000'
        }
    }
    return Sampler(net=model, **params)


def get_model(args, return_sampler=False):
    # Create model instance
    assert hasattr(sys.modules[__name__], args.model), f'Could not find model {args.model}'
    model_cls = getattr(sys.modules[__name__], args.model)
    model = model_cls().cuda()

    # Get statistics (count with batch size = 1)
    sample_input = torch.randn((1, *model_cls.sample_input_shape())).cuda()
    macs, params = profile(model, inputs=(sample_input, ), verbose=False)
    logging.info(f'Reference model {args.model} has {macs * 2 / 1e9:.5f}G FLOPs and {params / 1e6:.2f}M parameters')

    # Build mapping for usages
    sample_input = torch.randn((args.batch_size, *model_cls.sample_input_shape())).cuda()
    build_placeholder_mappings(model, sample_input)

    # Build sampler
    sampler = get_sampler(args, model) if (args.kas_replace_placeholder or return_sampler) else None

    # Replace kernel
    if args.kas_replace_placeholder is not None:
        logging.info(f'Replacing kernel with {args.kas_replace_placeholder} ...')
        cls_name = args.kas_replace_placeholder.capitalize() + 'Placeholder'
        assembled = getattr(placeholder, cls_name).impl(sampler.create_assembler())
        logging.debug(f'Assembled path: {assembled.convert_to_path(sampler)}')
        kernel_packs = sampler.realize(model, assembled, args.kas_replace_placeholder).construct_kernel_packs()
        sampler.replace(model, kernel_packs)

    if return_sampler:
        assert sampler
    return (model, sampler) if return_sampler else model
