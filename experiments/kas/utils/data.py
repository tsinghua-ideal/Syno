from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(args, data_path='~/data/'):
    if args.dataset == 'mnist':
        train_dataset = MNIST(root=data_path, train=True, download=True,
                              transform=transforms.Compose([
                                  # transforms.RandomAffine(
                                  #     degrees=10,
                                  #     translate=(0.1, 0.1),
                                  #     scale=(0.9, 1.1),
                                  #     shear=0.3,
                                  #     interpolation=transforms.InterpolationMode.BILINEAR
                                  #     ),
                                  transforms.ToTensor(),
                                  transforms.Normalize(args.mean, args.std),
                              ]))
        val_dataset = MNIST(root=data_path, train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(args.mean, args.std),
                            ]))

    elif args.dataset == 'cifar10':
        train_dataset = CIFAR10(root=data_path, train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(args.mean, args.std),
                                ]))

        val_dataset = CIFAR10(root=data_path, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(args.mean, args.std),
                              ]))

    else:
        raise NotImplementedError(
            f"Dataset {args.dataset} is not supported yet.")

    # construct data loader

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
        pin_memory_device="cuda"
    )
    validation_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
        pin_memory_device="cuda"
    )

    return train_data_loader, validation_data_loader

# Timm dataloader. 3 seconds slower for each epoch.
# train_data = create_dataset(
#     name='torch/cifar10',
#     root=data_path,
#     download=True,
#     split='train',
#     batch_size=args.batch_size,
# )
# validation_data = create_dataset(
#     name='torch/cifar10',
#     root=data_path,
#     download=True,
#     split='validation',
#     batch_size=args.batch_size,
# )

# Setup mixup / cutmix.
# collate_fn = None
# mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
# if mixup_active:
#     mixup_args = dict(
#         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
#         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
#         label_smoothing=args.smoothing, num_classes=args.num_classes)
#     collate_fn = FastCollateMixup(**mixup_args)

# train_data_loader = create_loader(
#     train_data,
#     input_size=args.input_size,
#     batch_size=args.batch_size,
#     is_training=True,
#     use_prefetcher=True,
#     no_aug=args.no_aug,
#     # re_prob=args.re_prob,
#     # re_mode=args.re_mode,
#     # re_count=args.re_count,
#     # re_split=args.re_split,
#     # scale=args.scale,
#     # ratio=args.ratio,
#     hflip=args.hflip,
#     vflip=args.vflip,
#     color_jitter=args.color_jitter,
#     # auto_augment=args.aa,
#     # interpolation=args.train_interpolation,
#     mean=args.mean,
#     std=args.std,
#     num_workers=args.num_workers,
#     # collate_fn=collate_fn,
#     pin_memory=args.pin_memory,
#     use_multi_epochs_loader=args.use_multi_epochs_loader
# )

# validation_data_loader = create_loader(
#     validation_data,
#     input_size=args.input_size,
#     batch_size=args.batch_size,
#     is_training=False,
#     use_prefetcher=True,
#     # interpolation=args.interpolation,
#     mean=args.mean,
#     std=args.std,
#     num_workers=args.num_workers,
#     # crop_pct=args.crop_pct,
#     pin_memory=args.pin_memory
# )
