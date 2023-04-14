import gc
import itertools
import torch
from torch import nn
import ptflops
import numpy as np

from KAS import Sampler, CodeGenOptions, MCTS, Modifier, KernelPack
import random

from copy import deepcopy
from base import dataset, device, log, models, parser, trainer


class model_backup():
    def __init__(self, model: nn.Module):
        self.model = deepcopy(model).cpu()

    def restore_model_params_and_replace(self, model, pack=None):
        """
        Restore model parameters and replace the selected parameters with pack.
        """
        model = deepcopy(self.model).to(args.device)
        if pack is not None:
            assert isinstance(pack, KernelPack), "pack is not valid!"
            Modifier.KernelReplace(model, pack.module, args.device)
        return model


if __name__ == '__main__':
    # Get arguments.
    logger = log.get_logger()
    args = parser.arg_parse()
    logger.info(f'Program arguments: {args}')

    # Check available devices and set distributed.
    logger.info(f'Initializing devices ...')
    device.initialize(args)
    assert not args.distributed, 'Search mode does not support distributed training'

    # Training utils.
    logger.info(f'Configuring model {args.model} ...')
    model = models.get_model(args, search_mode=True)
    train_loader, eval_loader = dataset.get_loaders(args)
    proxy_train_loader, proxy_eval_loader = dataset.get_loaders(
        args, proxy=True)

    # Load checkpoint.
    if args.kas_load_checkpoint:
        logger.info(f'Loading checkpoint from {args.kas_load_checkpoint}')
        checkpoint = torch.load(args.kas_load_checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Set up KAS randomness seed.
    logger.info(f'Configuring KAS Sampler ...')
    kas_sampler = Sampler(
        input_shape="[N,C_in,H,W]",
        output_shape="[N,C_out,H,W]",
        primary_specs=[],
        coefficient_specs=["s_1=2: 2", "k_1=3", "4"],
        seed=random.SystemRandom().randint(
            0, 0x7fffffff) if args.kas_seed == 'pure' else args.seed,
        depth=args.kas_depth,
        dim_lower=args.kas_min_dim,
        dim_upper=args.kas_max_dim,
        save_path=args.kas_sampler_save_dir,
        cuda=True,
        autoscheduler=CodeGenOptions.AutoScheduler.ComputeRoot
    )
    # mcts = MCTS(kas_sampler)

    # Initialization of search.
    backup_model = model_backup(model)

    # Search.
    current_best_score = 0
    round_range = range(
        args.kas_rounds) if args.kas_rounds > 0 else itertools.count()
    logger.info(
        f'Start KAS kernel search ({args.kas_rounds if args.kas_rounds else "infinite"} rounds)')
    for i in round_range:
        # Sample a new kernel.
        logger.info('Sampling a new kernel ...')
        g_macs, m_flops = 0, 0
        try:
            kernelPacks = kas_sampler.SampleKernel(model)
            backup_model.restore_model_params_and_replace(
                model, deepcopy(kernelPacks))
            g_macs, m_params = ptflops.get_model_complexity_info(model, args.input_size,
                                                                 as_strings=False, print_per_layer_stat=False)
            g_macs, m_params = g_macs / 1e9, m_params / 1e6
        except RuntimeError as ex:
            # Early exception: out of memory or timeout.
            logger.info(f'Exception: {ex}')
            log.save(args, None, None, None, {'exception': f'{ex}'})
            continue
        logger.info(f'Sampled kernel hash: {hash(kernelPacks)}')
        logger.info(f'MACs: {g_macs} G, params: {m_params} M')
        macs_not_satisfied = (args.kas_min_macs > 0 or args.kas_max_macs > 0) and \
                             (g_macs < args.kas_min_macs or g_macs >
                              args.kas_max_macs)
        params_not_satisfied = (args.kas_min_params > 0 or args.kas_max_params > 0) and \
                               (m_params < args.kas_min_params or m_params >
                                args.kas_max_params)
        if macs_not_satisfied or params_not_satisfied:
            logger.info(f'MACs ({args.kas_min_macs}, {args.kas_max_macs}) or '
                        f'params ({args.kas_min_params}, {args.kas_max_params}) '
                        f'requirements do not satisfy')
            continue

        # Train.
        proxy_score, train_metrics, eval_metrics, exception_info = 0, None, None, None
        try:
            if proxy_train_loader and proxy_eval_loader:
                logger.info('Training on proxy dataset ...')
                _, proxy_eval_metrics = \
                    trainer.train(args, model=model,
                                  train_loader=proxy_train_loader, eval_loader=proxy_eval_loader,
                                  search_mode=True, proxy_mode=True)
                assert len(proxy_eval_metrics) > 0
                best_epoch = 0
                for e in range(1, len(proxy_eval_metrics)):
                    if proxy_eval_metrics[e]['top1'] > proxy_eval_metrics[best_epoch]['top1']:
                        best_epoch = e
                proxy_score = proxy_eval_metrics[best_epoch]['top1']
                kernel_scales = proxy_eval_metrics[best_epoch]['kernel_scales']
                backup_model.restore_model_params_and_replace(
                    model, deepcopy(kernelPacks))
                gc.collect()
                logger.info(f'Proxy dataset score: {proxy_score}')
                if proxy_score < args.kas_proxy_threshold:
                    logger.info(
                        f'Under proxy threshold {args.kas_proxy_threshold}, skip main dataset training')
                    continue
                if len(kernel_scales) > 0 and args.kas_proxy_kernel_scale_limit > 0:
                    g_mean = np.exp(np.log(kernel_scales).mean())
                    if g_mean < args.kas_proxy_kernel_scale_limit or \
                       g_mean > 1 / args.kas_proxy_kernel_scale_limit:
                        logger.info(f'Breaking proxy scale limit {args.kas_proxy_kernel_scale_limit} '
                                    f'(gmean={g_mean}), '
                                    f'skip main dataset training')
                        continue
            logger.info('Training on main dataset ...')
            train_metrics, eval_metrics = \
                trainer.train(args, model=model,
                              train_loader=train_loader, eval_loader=eval_loader,
                              search_mode=True)
            score = max([item['top1'] for item in eval_metrics])
            logger.info(f'Solution score: {score}')
            if score > current_best_score:
                current_best_score = score
                logger.info(f'Current best score: {current_best_score}')
                if args.kas_weight_sharing:
                    try:
                        cpu_clone = deepcopy(model).cpu()
                        logger.info(f'Weight successfully shared')
                    except Exception as ex:
                        logger.warning(f'Failed to make weight shared: {ex}')
        except RuntimeError as ex:
            exception_info = f'{ex}'
            logger.warning(f'Exception: {exception_info}')

        # Save into logging directory.
        extra = {'proxy_score': proxy_score,
                 'g_macs': g_macs, 'm_params': m_params}
        if exception_info:
            extra['exception'] = exception_info
        log.save(args, kernelPacks, train_metrics, eval_metrics, extra)
