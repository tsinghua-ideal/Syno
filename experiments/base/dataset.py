import logging
from timm.data import create_dataset, create_loader


def get_dataloader(args):
    train_data = create_dataset(
        name=args.dataset,
        root=args.root,
        download=True,
        split='train',
        batch_size=args.batch_size,
    )

    train_dataloader = create_loader(
        train_data,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=True,
        no_aug=True,
        mean=args.mean,
        std=args.std,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    validation_data = create_dataset(
        name=args.dataset,
        root=args.root,
        download=True,
        split='valid',
        batch_size=args.batch_size,
    )

    validation_dataloader = create_loader(
        validation_data,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=True,
        mean=args.mean,
        std=args.std,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    # Prefetch all data into GPU
    if args.fetch_all_to_gpu:
        logging.info('Fetching all data into GPU ...')
        train_dataloader = [(image, label) for (image, label) in train_dataloader]
        validation_dataloader = [(image, label) for (image, label) in validation_dataloader]

    return train_dataloader, validation_dataloader
