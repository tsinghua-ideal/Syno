import logging
import numpy as np
import os, torch

from base import log, models, trainer, parser, dataset

from KAS import KernelLoader

kernel_directories = {
    # "attn": "results/gpt-session-v20240722/05762_17394058425718992829",
    "attn": "results/gpt-session-2023/05658_17539554982746112344",
    # "fc": "results/gpt-session-fc-v20240723/06251_2779362985208389457"
    # "fc": "results/gpt-session-fc-v20240723/06297_45523330860059992"
}

if __name__ == '__main__':
    log.setup()

    args = parser.arg_parse()

    logging.info('Loading dataset ...')
    train_dataloader, val_dataloader = dataset.get_dataloader(args)

    logging.info('Preparing model ...')
    sample_input = None
    if 'gcn' in args.model:
        sample_input = train_dataloader
    model = models.get_model(args, sample_input=sample_input)

    if "LOAD" in os.environ and os.environ["LOAD"]:
        kernel_loaders = {
            k: KernelLoader.from_directory(os.path.join(
                    kernel_directory, "kernel_scheduler_dir"
            ))
            for k, kernel_directory in kernel_directories.items()
        }

        flops, params = model.profile(seq_len=args.gpt_seq_len)
        model.load_kernels(
            kernel_loaders,
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

    logging.info('Start training ...')
    losses = trainer.train(model, train_dataloader, val_dataloader, args)

    if 'gpt' in args.model or 'rwkv' in args.model:
        if args.gpt_loss_output:
            with open(args.gpt_loss_output, "w") as f:
                f.write(f"{losses}")
        
        losses = list(map(lambda t: t[1], losses))
        assert len(losses) >= 1
        len_not_avg = max(int(len(losses) * 0.8), 1)
        loss = np.mean(losses[len_not_avg - 1:])
        logging.debug(f"Meaned loss of last 20%: {loss}")

