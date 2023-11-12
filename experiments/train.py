import logging
import numpy as np

from base import log, models, trainer, parser, dataset


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

    logging.info('Start training ...')
    losses = trainer.train(model, train_dataloader, val_dataloader, args)

    if 'gpt' in args.model:
        if args.gpt_loss_output:
            with open(args.gpt_loss_output, "w") as f:
                f.write(f"{losses}")
        
        losses = list(map(lambda t: t[1], losses))
        assert len(losses) >= 1
        len_not_avg = max(int(len(losses) * 0.8), 1)
        loss = np.mean(losses[len_not_avg - 1:])
        logging.debug(f"Meaned loss of last 20%: {loss}")
