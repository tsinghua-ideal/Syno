from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from timm.data import create_dataset, create_loader


def get_dataloader(args, data_path='~/data/'):
    dataset = CIFAR10(root=data_path, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(
                              (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      ]))
    train_data, validation_data = random_split(dataset, [40000, 10000])
    train_data = create_dataset(
        name='torch/cifar10',
        root=data_path,
        download=True,
        split='train',
        batch_size=args.batch_size,
    )
    validation_data = create_dataset(
        name='torch/cifar10',
        root=data_path,
        download=True,
        split='validation',
        batch_size=args.batch_size,
    )

    train_data_loader = create_loader(
        train_data,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=True,
        no_aug=args.no_aug,
        re_prob=args.re_prob,
        re_mode=args.re_mode,
        re_count=args.re_count,
        re_split=args.re_split,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=args.train_interpolation,
        mean=(0.49139968, 0.48215827, 0.44653124),
        std=(0.24703233, 0.24348505, 0.26158768),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )

    validation_data_loader = create_loader(
        validation_data,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=True,
        # interpolation=args.interpolation,
        mean=(0.49139968, 0.48215827, 0.44653124),
        std=(0.24703233, 0.24348505, 0.26158768),
        num_workers=args.num_workers,
        crop_pct=args.crop_pct,
        pin_memory=args.pin_memory
    )

    # construct data loader

    # train_data_loader = DataLoader(
    #     train_data,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     shuffle=True,
    #     drop_last=True,
    #     pin_memory=True,
    #     pin_memory_device="cuda:0"
    # )
    # validation_data_loader = DataLoader(
    #     validation_data,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     shuffle=False,
    #     drop_last=True,
    #     pin_memory=True,
    #     pin_memory_device="cuda:0"
    # )

    return train_data_loader, validation_data_loader
