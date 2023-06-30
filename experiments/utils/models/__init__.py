import random
import sys
import torch
from thop import profile
from KAS import Sampler
from KAS.Bindings import CodeGenOptions

from . import kas_impl
from .conv_net import ConvNet, KASConvNet, KASGrayConvNet
from .dense_net import FCNet, KASFCNet


def get_model_input_size(args):
    assert hasattr(sys.modules[__name__], args.model), f'Could not find model {args.model}'
    model_cls = getattr(sys.modules[__name__], args.model)
    return model_cls.sample_input_shape()


def get_model_and_sampler(args):
    # Create model instance
    assert hasattr(sys.modules[__name__], args.model), f'Could not find model {args.model}'
    model_cls = getattr(sys.modules[__name__], args.model)
    model = model_cls().cuda()

    # Get statistics (count with batch size = 1)
    sample_input = torch.randn((1, *model_cls.sample_input_shape())).cuda()
    macs, params = profile(model, inputs=(sample_input, ), verbose=False)
    print(f'Reference model {args.model} has {macs * 2 / 1e9:.5f}G FLOPs and {params / 1e6:.2f}M parameters')

    # KAS sampler
    model_params = model_cls.sampler_parameters()
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
        'autoscheduler': CodeGenOptions.AutoScheduler.Anderson2021
    }
    sampler = Sampler(net=model, **params)

    # Replace kernel
    if args.kas_replace_placeholder is not None:
        print(f'Replacing kernel with {args.kas_replace_placeholder} ...')
        assembled = getattr(kas_impl, args.kas_replace_placeholder)(sampler.create_assembler())
        print(f'Assembled path: {assembled.convert_to_path(sampler)}')
        kernel_packs, total_flops = sampler.realize(model, assembled, args.kas_replace_placeholder)
        sampler.replace(model, kernel_packs)
        # TODO: print new FLOPs and parameters

    return model, sampler
