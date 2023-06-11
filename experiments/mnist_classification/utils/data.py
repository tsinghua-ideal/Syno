from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_dataloader(args, data_path='~/data/'):
    train_dataset = MNIST(root=data_path, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(args.mean, args.std),
                          ]))
    val_dataset = MNIST(root=data_path, train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(args.mean, args.std),
                        ]))

    # construct data loader

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.pin_memory
    )
    validation_data_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory
    )

    return train_data_loader, validation_data_loader
