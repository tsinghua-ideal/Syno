import torch as ch
from torch.amp import GradScaler
from torch.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist

ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

from torchvision import models
import torchmetrics
import numpy as np

from .quant import quantize

import os, sys
import time, datetime
import json
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser
import logging
from tqdm import tqdm

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    Squeeze,
    NormalizeImage,
    RandomHorizontalFlip,
    ToTorchImage,
    ImageMixup,
    LabelMixup,
)
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.fields.basics import IntDecoder

Section("model", "model details").params(
    arch=Param(And(str, OneOf(models.__dir__())), default="resnet18"),
    pretrained=Param(int, "is pretrained? (1/0)", default=0),
)

Section("resolution", "resolution scheduling").params(
    min_res=Param(int, "the minimum (starting) resolution", default=160),
    max_res=Param(int, "the maximum (starting) resolution", default=160),
    end_ramp=Param(int, "when to stop interpolating resolution", default=0),
    start_ramp=Param(int, "when to start interpolating resolution", default=0),
)

Section("data", "data related stuff").params(
    train_dataset=Param(str, ".dat file to use for training", required=True),
    val_dataset=Param(str, ".dat file to use for validation", required=True),
    num_workers=Param(int, "The number of workers", required=True),
    in_memory=Param(int, "does the dataset fit in memory? (1/0)", required=True),
    mixup_alpha=Param(float, "alpha for mixup", default=0.0),
    raug_mag=Param(int, "magnitude of RandAug", default=0),
    raug_layer=Param(int, "layers of RandAug", default=0),
)

Section("lr", "lr scheduling").params(
    step_ratio=Param(float, "learning rate step ratio", default=0.1),
    step_length=Param(int, "learning rate step length", default=30),
    warmup_epochs=Param(int, "Warmup Epochs", default=0),
    lr_schedule_type=Param(OneOf(["step", "cyclic", "cosine"]), default="cyclic"),
    lr=Param(float, "learning rate", default=0.5),
    lr_peak_epoch=Param(int, "Epoch at which LR peaks", default=2),
)

Section("logging", "how to log stuff").params(
    folder=Param(str, "log location", required=True),
    log_level=Param(int, "0 if only at end 1 otherwise", default=1),
)

Section("validation", "Validation parameters stuff").params(
    batch_size=Param(int, "The batch size for validation", default=512),
    resolution=Param(int, "final resized validation image size", default=224),
    lr_tta=Param(int, "should do lr flipping/avging at test time", default=1),
)

Section("training", "training hyper param stuff").params(
    eval_only=Param(int, "eval only?", default=0),
    batch_size=Param(int, "The batch size", default=512),
    optimizer=Param(And(str, OneOf(["sgd", "adam"])), "The optimizer", default="sgd"),
    momentum=Param(float, "SGD momentum", default=0.9),
    weight_decay=Param(float, "weight decay", default=4e-5),
    epochs=Param(int, "number of epochs", default=30),
    label_smoothing=Param(float, "label smoothing parameter", default=0.1),
    distributed=Param(int, "is distributed?", default=0),
    use_blurpool=Param(int, "use blurpool?", default=0),
    enable_amp=Param(int, "use amp autocast?", default=1),
)

Section("dist", "distributed training options").params(
    world_size=Param(int, "number gpus", default=1),
    address=Param(str, "address", default="localhost"),
    port=Param(str, "port", default="12355"),
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256


@param("lr.lr")
@param("lr.step_ratio")
@param("lr.step_length")
@param("training.epochs")
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr


@param("lr.lr")
@param("lr.warmup_epochs")
@param("training.epochs")
def get_cosine_lr(
    epoch,
    lr,
    warmup_epochs,
    epochs,
):
    if epoch >= epochs:
        return 0

    # Cosine decay
    learning_rate = (
        0.5
        * lr
        * (1 + np.cos(np.pi * (epoch - warmup_epochs) / float(epochs - warmup_epochs)))
    )

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = lr * (epoch / warmup_epochs)

    learning_rate = np.where(epoch < warmup_epochs, warmup_lr, learning_rate)
    return learning_rate


@param("lr.lr")
@param("training.epochs")
@param("lr.lr_peak_epoch")
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


class ImageNetTrainer:
    @param("training.distributed")
    def __init__(self, model, folder, batch_size, gpu, eval_only, quant, distributed):
        self.all_params = get_current_config()
        self.gpu = gpu

        self.uid = str(uuid4())

        if distributed:
            self.setup_distributed()

        self.train_loader = None if eval_only else self.create_train_loader(batch_size=batch_size)
        self.val_loader = self.create_val_loader(batch_size=batch_size)
        self.model, self.scaler = self.create_model_and_scaler(model)
        self.create_optimizer()
        self.initialize_logger(folder)

        if quant:
            quantize(self.model, self.val_loader, 1)

    @param("dist.address")
    @param("dist.port")
    @param("dist.world_size")
    def setup_distributed(self, address, port, world_size):
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)
        ch.cuda.empty_cache()

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param("lr.lr_schedule_type")
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            "cyclic": get_cyclic_lr,
            "step": get_step_lr,
            "cosine": get_cosine_lr,
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param("resolution.min_res")
    @param("resolution.max_res")
    @param("resolution.end_ramp")
    @param("resolution.start_ramp")
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param("training.momentum")
    @param("training.optimizer")
    @param("training.weight_decay")
    @param("training.label_smoothing")
    def create_optimizer(self, momentum, optimizer, weight_decay, label_smoothing):
        assert optimizer in ["sgd", "adam"], f"{optimizer} is not valid"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ("bn" in k)]
        other_params = [v for k, v in all_params if not ("bn" in k)]
        param_groups = [
            {"params": bn_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": weight_decay},
        ]

        if optimizer == "sgd":
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        else:
            self.optimizer = ch.optim.Adam(param_groups, lr=1)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param("data.train_dataset")
    @param("data.num_workers")
    @param("training.distributed")
    @param("data.in_memory")
    @param("data.mixup_alpha")
    @param("data.raug_mag")
    @param("data.raug_layer")
    def create_train_loader(
        self,
        batch_size,
        train_dataset,
        num_workers,
        distributed,
        in_memory,
        mixup_alpha,
        raug_mag,
        raug_layer,
    ):
        this_device = f"cuda:{self.gpu}"
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            # RandAugment(raug_layer, raug_mag),
            # ImageMixup(mixup_alpha, same_lambda=False),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            # LabelMixup(mixup_alpha, same_lambda=False),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True),
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            seed=0,
            os_cache=in_memory,
            drop_last=True,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )

        return loader

    @param("data.val_dataset")
    @param("data.num_workers")
    @param("validation.resolution")
    @param("training.distributed")
    def create_val_loader(
        self, batch_size, val_dataset, num_workers, resolution, distributed
    ):
        this_device = f"cuda:{self.gpu}"
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True),
        ]

        loader = Loader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            seed=0,
            drop_last=True,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )
        return loader

    @param("training.epochs")
    @param("logging.log_level")
    def train(self, epochs, log_level):
        for epoch in range(epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            start_train = time.time()
            train_loss = self.train_loop(epoch)
            train_time = time.time() - start_train

            if log_level > 0:
                extra_dict = {
                    "train_loss": train_loss,
                    "epoch": epoch,
                    "train_time": train_time,
                }

                self.eval_and_log(extra_dict)
                
            if (epoch + 1) % 10 == 0 and self.gpu == 0:
                ch.save(self.model.state_dict(), self.log_folder / f"weights_{epoch + 1}.pt")

        stats = self.eval_and_log({"epoch": epoch})
        # if self.gpu == 0:
        #     ch.save(self.model.state_dict(), self.log_folder / "final_weights.pt")
        return stats["top_1"]

    def eval_and_log(self, extra_dict={}):
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        if self.gpu == 0:
            self.log(
                dict(
                    {
                        "current_lr": self.optimizer.param_groups[0]["lr"],
                        "top_1": stats["top_1"],
                        "top_5": stats["top_5"],
                        "val_time": val_time,
                    },
                    **extra_dict,
                )
            )
            if "train_loss" in extra_dict:
                logging.info(
                    f"Epoch [{extra_dict['epoch'] + 1}], train loss: {extra_dict['train_loss']}, test accuracy: top_1={stats['top_1']}, top_5={stats['top_5']}, training time: {extra_dict['train_time']}, validation time: {val_time}"
                )
            else:
                logging.info(
                    f"Epoch [{extra_dict['epoch'] + 1}], test accuracy: top_1={stats['top_1']}, top_5={stats['top_5']}, validation time: {val_time}"
                )

        return stats

    @param("training.distributed")
    @param("training.use_blurpool")
    def create_model_and_scaler(self, model, distributed, use_blurpool):
        scaler = GradScaler("cuda")

        def apply_blurpool(mod: ch.nn.Module):
            for name, child in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (
                    np.max(child.stride) > 1 and child.in_channels >= 16
                ):
                    setattr(mod, name, BlurPoolConv2d(child))
                else:
                    apply_blurpool(child)

        if use_blurpool:
            apply_blurpool(model)

        try:
            model = model.to(memory_format=ch.channels_last)
        except:
            logging.info("Converting to ch.channels_last failed. ")
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler

    @param("logging.log_level")
    @param("training.enable_amp")
    def train_loop(self, epoch, log_level, enable_amp):
        model = self.model
        model.train()
        losses = []

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = self.train_loader
        for ix, (images, target) in enumerate(iterator):
            ### Training start
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lrs[ix]

            self.optimizer.zero_grad(set_to_none=True)
            if enable_amp:
                with autocast("cuda"):
                    output = self.model(images)
                    loss_train = self.loss(output, target)

                    self.scaler.scale(loss_train).backward()
                    self.scaler.unscale_(self.optimizer)
                    ch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                output = self.model(images)
                loss_train = self.loss(output, target)
                loss_train.backward()
                ch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
            ### Training end

            ### Logging start
            if log_level > 0:
                losses.append(loss_train.detach())

                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.3f}')

                names = ["ep", "iter", "shape", "lrs"]
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ["loss"]
                    values += [f"{loss_train.item():.3f}"]

                msg = ", ".join(f"{n}={v}" for n, v in zip(names, values))
                # iterator.set_description(msg)
                if ix % 100 == 0:
                    self.log({n: str(v) for n, v in zip(names, values)})
            ### Logging end

        return (sum(losses) / len(losses)).item()

    @param("validation.lr_tta")
    def val_loop(self, lr_tta):
        model = self.model
        model.eval()

        with ch.no_grad():
            for images, target in self.val_loader:
                output = self.model(images)
                if lr_tta:
                    output += self.model(ch.flip(images, dims=[3]))

                for k in ["top_1", "top_5"]:
                    self.val_meters[k](output, target)

                loss_val = self.loss(output, target)
                self.val_meters["loss"](loss_val)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats
        
    def initialize_logger(self, folder):
        self.val_meters = {
            "top_1": torchmetrics.Accuracy(task="multiclass", num_classes=1000).to(
                self.gpu
            ),
            "top_5": torchmetrics.Accuracy(
                task="multiclass", num_classes=1000, top_k=5
            ).to(self.gpu),
            "loss": MeanScalarMetric().to(self.gpu),
        }

        if self.gpu == 0:
            folder = (Path(folder) / str(self.uid)).absolute()
            folder.mkdir(parents=True)

            self.log_folder = folder
            self.start_time = time.time()

            print(f"=> Logging in {self.log_folder}")
            params = {
                ".".join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(folder / "params.json", "w+") as handle:
                json.dump(params, handle)

    def log(self, content):
        if self.gpu != 0:
            return
        cur_time = time.time()
        print(
            f"[{str(datetime.timedelta(seconds=cur_time - self.start_time))}] => Log: {content}"
        )
        sys.stdout.flush()
        with open(self.log_folder / "log", "a+") as fd:
            fd.write(
                json.dumps(
                    {
                        "timestamp": cur_time,
                        "relative_time": cur_time - self.start_time,
                        **content,
                    }
                )
                + "\n"
            )
            fd.flush()

    @classmethod
    @param("training.distributed")
    @param("dist.world_size")
    def launch_from_args(
        cls, model, folder, batch_size, eval_only, quantize, config_file, distributed, world_size
    ):
        ch.backends.cudnn.benchmark = True
        ch.backends.cuda.matmul.allow_tf32 = True
        ch.backends.cudnn.allow_tf32 = True
        if distributed:
            ch.multiprocessing.spawn(
                cls._exec_wrapper,
                args=(model, folder, batch_size, eval_only, quantize, config_file),
                nprocs=world_size,
                join=True,
            )
        else:
            return cls.exec(model, folder, batch_size, 0, eval_only, quantize)

    @classmethod
    def _exec_wrapper(cls, i, model, folder, batch_size, eval_only, quantize, config_file, **kwargs):
        make_config(config_file, quiet=True)
        cls.exec(model, folder, batch_size, i, eval_only, quantize, **kwargs)

    @classmethod
    @param("training.distributed")
    # @param("training.eval_only")
    def exec(cls, model, folder, batch_size, gpu, eval_only, quantize, distributed):
        trainer = cls(model=model, folder=folder, batch_size=batch_size, gpu=gpu, eval_only=eval_only, quant=quantize)
        if eval_only:
            trainer.eval_and_log({'epoch': 89})
        else:
            accuracy = trainer.train()

        if distributed:
            trainer.cleanup_distributed()

        if not eval_only:
            return accuracy


# Utils
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state("sum", default=ch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=ch.tensor(0), dist_reduce_fx="sum")

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count


def make_config(config_file, quiet=False):
    config = get_current_config()
    config.collect_config_file(config_file)
    config.validate(mode="stderr")
    if not quiet:
        config.summary()
