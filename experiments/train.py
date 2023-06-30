from utils import models, trainer, parser, dataset


if __name__ == '__main__':
    args = parser.arg_parse()

    print('Preparing model ...')
    model, _ = models.get_model_and_sampler(args)

    print('Loading dataset ...')
    train_loader, val_loader = dataset.get_dataloader(args)

    print('Start training ...')
    trainer.train(model, train_loader, val_loader, args)
