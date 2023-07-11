import logging

from base import log, models, trainer, parser, dataset


if __name__ == '__main__':
    log.setup()

    args = parser.arg_parse()

    logging.info('Preparing model ...')
    model = models.get_model(args)

    logging.info('Loading dataset ...')
    train_dataloader, val_dataloader = dataset.get_dataloader(args)

    logging.info('Start training ...')
    trainer.train(model, train_dataloader, val_dataloader, args)
