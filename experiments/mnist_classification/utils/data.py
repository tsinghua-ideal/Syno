from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader


def get_dataloader(args, data_path='~/data/'):
    mnist_dataset = MNIST(root=data_path, train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(0, 1),
                          ]))
    train_data, validation_data = random_split(mnist_dataset, [50000, 10000])

    print("train data", len(train_data))
    print("val data", len(validation_data))

    # construct data loader

    train_data_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True, 
        drop_last=True,
        pin_memory=True,
        pin_memory_device="cuda:0"
    )
    validation_data_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False, 
        drop_last=True,
        pin_memory=True,
        pin_memory_device="cuda:0"
    )

    return train_data_loader, validation_data_loader
