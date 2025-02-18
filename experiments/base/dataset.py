import logging
import os, sys
import torch
import torch.nn.functional as F
from datasets import load_dataset, IterableDataset
from functools import partial
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageNet
from torchvision.transforms import transforms


def make_random_square_masks(inputs, mask_size):
    if mask_size == 0:
        return None
    is_even = int(mask_size % 2 == 0)
    in_shape = inputs.shape

    mask_center_y = torch.empty(
        in_shape[0], dtype=torch.long, device=inputs.device
    ).random_(mask_size // 2 - is_even, in_shape[-2] - mask_size // 2 - is_even)
    mask_center_x = torch.empty(
        in_shape[0], dtype=torch.long, device=inputs.device
    ).random_(mask_size // 2 - is_even, in_shape[-1] - mask_size // 2 - is_even)

    to_mask_y_dists = torch.arange(in_shape[-2], device=inputs.device).view(
        1, 1, in_shape[-2], 1
    ) - mask_center_y.view(-1, 1, 1, 1)
    to_mask_x_dists = torch.arange(in_shape[-1], device=inputs.device).view(
        1, 1, 1, in_shape[-1]
    ) - mask_center_x.view(-1, 1, 1, 1)

    to_mask_y = (to_mask_y_dists >= (-(mask_size // 2) + is_even)) * (
        to_mask_y_dists <= mask_size // 2
    )
    to_mask_x = (to_mask_x_dists >= (-(mask_size // 2) + is_even)) * (
        to_mask_x_dists <= mask_size // 2
    )

    final_mask = to_mask_y * to_mask_x
    return final_mask


@torch.no_grad()
def batch_cutout(inputs, patch_size):
    cutout_batch_mask = make_random_square_masks(inputs, patch_size)
    inputs = torch.where(cutout_batch_mask, torch.zeros_like(inputs), inputs)
    return inputs


@torch.no_grad()
def batch_crop(inputs, crop_size):
    crop_mask_batch = make_random_square_masks(inputs, crop_size)
    cropped_batch = torch.masked_select(inputs, crop_mask_batch).view(
        inputs.shape[0], inputs.shape[1], crop_size, crop_size
    )
    return cropped_batch


@torch.no_grad()
def batch_flip_lr(batch_images, flip_chance=0.5):
    return torch.where(
        torch.rand_like(batch_images[:, 0, 0, 0].view(-1, 1, 1, 1)) < flip_chance,
        torch.flip(batch_images, (-1,)),
        batch_images,
    )


@torch.no_grad()
def get_batches(data_dict, key, batch_size, crop_size):
    num_epoch_examples = len(data_dict["images"])
    shuffled = torch.randperm(num_epoch_examples, device="cuda")

    if key == "train":
        images = batch_crop(data_dict["images"], crop_size)
        images = batch_flip_lr(images)
        images = batch_cutout(images, patch_size=3)
    else:
        images = data_dict["images"]
    labels = data_dict["labels"]

    for idx in range(num_epoch_examples // batch_size):
        if not (idx + 1) * batch_size > num_epoch_examples:
            x, y = images.index_select(
                0, shuffled[idx * batch_size : (idx + 1) * batch_size]
            ), labels.index_select(
                0, shuffled[idx * batch_size : (idx + 1) * batch_size]
            )
            x = F.interpolate(x, size=(224, 224), mode="bilinear")
            yield x, y


class FuncDataloader:
    def __init__(self, func):
        self.func = func

    def __iter__(self):
        return self.func()


def get_dataloader(args):
    if "Cora" in args.dataset:
        return get_gnn_dataloader(args)
    if "lm1b" in args.dataset:
        return get_gpt_dataloader(args)
    if "imagenet" in args.dataset:
        return None, None

    # Get tensors
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_name = str(args.dataset).upper()
    assert hasattr(
        sys.modules[__name__], dataset_name
    ), f"Could not find dataset {dataset_name}"
    func = getattr(sys.modules[__name__], dataset_name)
    train_data = func(root=args.root, download=True, train=True, transform=transform)
    eval_data = func(root=args.root, download=False, train=False, transform=transform)

    # Get dataloaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=len(train_data),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=len(eval_data),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
    )

    # Compose to datasets
    train_dataset, eval_dataset = {}, {}
    train_dataset["images"], train_dataset["labels"] = [
        item.cuda(non_blocking=True) for item in next(iter(train_dataloader))
    ]
    eval_dataset["images"], eval_dataset["labels"] = [
        item.cuda(non_blocking=True) for item in next(iter(eval_dataloader))
    ]

    # Normalize images
    std, mean = torch.std_mean(train_dataset["images"], dim=(0, 2, 3))

    def batch_normalize_images(input_images, mean, std):
        return (input_images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

    batch_normalize_images = partial(batch_normalize_images, mean=mean, std=std)
    train_dataset["images"] = batch_normalize_images(train_dataset["images"])
    eval_dataset["images"] = batch_normalize_images(eval_dataset["images"])
    logging.info(f"Std: {std}, mean: {mean}")

    # Padding
    assert (
        train_dataset["images"].shape[-1] == train_dataset["images"].shape[-2]
    ), "Images must be square"
    crop_size = train_dataset["images"].shape[-1]
    train_dataset["images"] = F.pad(train_dataset["images"], (2,) * 4, "reflect")

    return FuncDataloader(
        partial(
            get_batches,
            data_dict=train_dataset,
            key="train",
            batch_size=args.batch_size,
            crop_size=crop_size,
        )
    ), FuncDataloader(
        partial(
            get_batches,
            data_dict=eval_dataset,
            key="eval",
            batch_size=args.batch_size,
            crop_size=crop_size,
        )
    )


class ConstantLengthDataset(IterableDataset):
    def __init__(
        self, tokenizer, dataset, infinite=False, seq_length=1024, chars_per_token=3.6
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length + 1
        # TODO: fix factor and prefetch
        self.input_characters = seq_length * chars_per_token * 128
        self.epoch = 0
        self.infinite = infinite

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)["text"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logging.info(f"Current epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids).unsqueeze(0)


def get_gpt_dataloader(args):
    from transformers import GPT2Tokenizer
    # TODO: add multiple workers
    logging.info(f"Loading GPT dataset {args.dataset} ...")
    dataset = load_dataset(str(args.dataset))
    tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_tokenizer)
    return ConstantLengthDataset(
        tokenizer, dataset["train"], infinite=True, seq_length=args.gpt_seq_len
    ), ConstantLengthDataset(
        tokenizer, dataset["test"], infinite=False, seq_length=args.gpt_seq_len
    )


def get_imagenet_dataloader(args):
    from timm.data import FastCollateMixup, create_loader
    from torchvision.datasets import ImageFolder

    dataset_train = ImageFolder(
        root=os.path.join(args.root, "train"),
    )
    dataset_eval = ImageFolder(
        root=os.path.join(args.root, "val"),
    )

    # Setup mixup / cutmix.
    collate_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        collate_fn = FastCollateMixup(**mixup_args)

    # Create data loaders w/ augmentation pipeline.
    train_loader = create_loader(
        dataset_train,
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
        # num_aug_repeats=aug_repeats,
        # num_aug_splits=0,
        interpolation=args.train_interpolation,
        mean=args.mean,
        std=args.std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
    )

    eval_loader = create_loader(
        dataset_eval,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation=args.interpolation,
        mean=args.mean,
        std=args.std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        crop_pct=args.crop_pct,
        pin_memory=args.pin_memory,
    )

    return train_loader, eval_loader


def get_gnn_dataloader(args):
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root=args.root, name=args.dataset)[0].cuda()
    return dataset, dataset
