import logging
import sys
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.transforms import transforms


def make_random_square_masks(inputs, mask_size):
    if mask_size == 0:
        return None
    is_even = int(mask_size % 2 == 0)
    in_shape = inputs.shape

    mask_center_y = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size // 2 - is_even, in_shape[-2] - mask_size // 2 - is_even)
    mask_center_x = torch.empty(in_shape[0], dtype=torch.long, device=inputs.device).random_(mask_size // 2 - is_even, in_shape[-1] - mask_size // 2 - is_even)

    to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(1, 1, in_shape[-2], 1) - mask_center_y.view(-1, 1, 1, 1)
    to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(1, 1, 1, in_shape[-1]) - mask_center_x.view(-1, 1, 1, 1)

    to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (to_mask_y_dists <= mask_size // 2)
    to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (to_mask_x_dists <= mask_size // 2)

    final_mask = to_mask_y * to_mask_x
    return final_mask


def batch_cutout(inputs, patch_size):
    with torch.no_grad():
        cutout_batch_mask = make_random_square_masks(inputs, patch_size)
        inputs = torch.where(cutout_batch_mask, torch.zeros_like(inputs), inputs)
        return inputs


def batch_crop(inputs, crop_size):
    with torch.no_grad():
        crop_mask_batch = make_random_square_masks(inputs, crop_size)
        cropped_batch = torch.masked_select(inputs, crop_mask_batch).view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)
        return cropped_batch


def batch_flip_lr(batch_images, flip_chance=.5):
    with torch.no_grad():
        return torch.where(torch.rand_like(batch_images[:, 0, 0, 0].view(-1, 1, 1, 1)) < flip_chance, torch.flip(batch_images, (-1,)), batch_images)


@torch.no_grad()
def get_batches(data_dict, key, batch_size, crop_size):
    num_epoch_examples = len(data_dict['images'])
    shuffled = torch.randperm(num_epoch_examples, device='cuda')

    if key == 'train':
        images = batch_crop(data_dict['images'], crop_size)
        images = batch_flip_lr(images)
        images = batch_cutout(images, patch_size=3)
    else:
        images = data_dict['images']
    labels = data_dict['labels']

    for idx in range(num_epoch_examples // batch_size):
        if not (idx + 1) * batch_size > num_epoch_examples:
            yield images.index_select(0, shuffled[idx * batch_size:(idx + 1) * batch_size]), \
                  labels.index_select(0, shuffled[idx * batch_size:(idx + 1) * batch_size])


class FuncDataloader():
    def __init__(self, func):
        self.func = func

    def __iter__(self):
        return self.func()


def get_dataloader(args):
    # Get tensors
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_name = str(args.dataset).upper()
    assert hasattr(sys.modules[__name__], dataset_name), f'Could not find dataset {args.model}'
    func = getattr(sys.modules[__name__], dataset_name)
    train_data = func(root=args.root, download=True, train=True, transform=transform)
    eval_data = func(root=args.root, download=False, train=False, transform=transform)

    # Get dataloaders
    train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=False)
    eval_dataloader = DataLoader(eval_data, batch_size=len(eval_data), shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=False)

    # Compose to datasets
    train_dataset, eval_dataset = {}, {}
    train_dataset['images'], train_dataset['labels'] = [item.cuda(non_blocking=True) for item in next(iter(train_dataloader))]
    eval_dataset['images'],  eval_dataset['labels']  = [item.cuda(non_blocking=True) for item in next(iter(eval_dataloader)) ]

    # Normalize images
    std, mean = torch.std_mean(train_dataset['images'], dim=(0, 2, 3))
    def batch_normalize_images(input_images, mean, std):
        return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    
    batch_normalize_images = partial(batch_normalize_images, mean=mean, std=std)
    train_dataset['images'] = batch_normalize_images(train_dataset['images'])
    eval_dataset['images']  = batch_normalize_images(eval_dataset['images'])
    logging.debug(f'Std: {std}, mean: {mean}')

    # Padding
    assert train_dataset['images'].shape[-1] == train_dataset['images'].shape[-2], 'Images must be square'
    crop_size = train_dataset['images'].shape[-1]
    train_dataset['images'] = F.pad(train_dataset['images'], (2, ) * 4, 'reflect')

    return FuncDataloader(partial(get_batches, data_dict=train_dataset, key='train', batch_size=args.batch_size, crop_size=crop_size)), \
        FuncDataloader(partial(get_batches, data_dict=eval_dataset, key='eval', batch_size=args.batch_size, crop_size=crop_size))
