import json
import logging
import torch
import time
from torch import nn
from typing import Tuple, List
from timm.utils import AverageMeter
from transformers import GPT2Tokenizer
from KAS import Placeholder, init_weights

from .loss import get_loss_func, get_gnn_loss_func
from .optim import get_optimizer, get_gpt_optimizer, get_gnn_optimizer
from .sche import get_schedule
from .models import KASModel


def torch_opt_on():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train(
    model,
    train_dataloader,
    val_dataloader,
    args,
    init_weight=True,
    use_bf16_train=False,
    use_bf16_test=False,
) -> List[float]:
    if "gpt" in args.model:
        return train_gpt(model, train_dataloader, val_dataloader, args)
    if "gcn" in args.model:
        return train_gnn(model, train_dataloader, val_dataloader, args)

    assert isinstance(model, nn.Module)

    torch_opt_on()
    assert torch.cuda.is_available(), "CUDA is not supported."
    model.cuda()
    if isinstance(model, KASModel):
        model.remove_thop_hooks()
        if init_weight:
            model.initialize_weights()

    # Loss, optimizer and scheduler
    loss_func = get_loss_func(args)
    optimizer = get_optimizer(model, args)
    scheduler, sched_epochs = get_schedule(optimizer, args)

    val_accuracy = []
    num_updates = 0
    milestones = dict()
    if args.prune_milestones:
        with open(args.prune_milestones) as f:
            milestones = json.load(f)
        logging.info(f"Milestones loaded: {milestones}")

    for epoch in range(sched_epochs):
        # Train
        start_time = time.time()
        if use_bf16_train:
            model.bfloat16()
        else:
            model.float()
        model.train()
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        for i, (image, label) in enumerate(train_dataloader):
            # Forward
            image, label = image.cuda(), label.cuda()
            total += label.size(0)
            logits: torch.Tensor = (
                model(image.bfloat16()).float() if use_bf16_train else model(image)
            )
            loss = loss_func(logits, label)

            # Statistic
            pred = torch.argmax(logits, 1)
            correct += torch.sum(pred == label).item()

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=args.grad_norm_clip, norm_type=2
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Update loss and scheduler
            num_updates += 1
            loss_meter.update(loss.item(), image.size(0))
            scheduler.step_update(num_updates=num_updates)
        scheduler.step(epoch=epoch + 1)
        elapsed_train_time = time.time() - start_time

        train_accuracy = correct / total

        # Valiation
        start_time = time.time()
        if use_bf16_test:
            model.bfloat16()
        else:
            model.float()
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (image, label) in enumerate(val_dataloader):
                image, label = image.cuda(), label.cuda()
                total += label.size(0)

                # A hack for the last batch
                if image.size(0) != args.batch_size:
                    shape = (args.batch_size - image.size(0), *image.size()[1:])
                    image = torch.cat(
                        [image, torch.zeros(shape, dtype=image.dtype).cuda()], dim=0
                    )
                    shape = (args.batch_size - label.size(0), *label.size()[1:])
                    label = torch.cat(
                        [label, torch.zeros(shape, dtype=label.dtype).cuda() - 1], dim=0
                    )

                # Inference
                logits = model(image.bfloat16()) if use_bf16_test else model(image)

                # Statistic
                pred = torch.argmax(logits, 1)
                correct += torch.sum(pred == label).item()
            val_accuracy.append(correct / total)
        elapsed_valid_time = time.time() - start_time
        logging.info(
            f"Epoch [{epoch + 1}/{sched_epochs}], train loss: {loss_meter.avg}, train_accuracy: {train_accuracy}, test accuracy: {correct / total}, training time: {elapsed_train_time}, validation time: {elapsed_valid_time}"
        )
        if (
            epoch > 0
            and args.kas_inference_time_limit
            and elapsed_train_time > args.kas_inference_time_limit
        ):
            logging.info(
                f"Inference time limit reached ({elapsed_train_time}s currently), stopping training ..."
            )
            break

        if (
            str(epoch + 1) in milestones
            and max(val_accuracy) < milestones[str(epoch + 1)]
        ):
            logging.info(f"Accuracy too low, pruning ...")
            break

    logging.info(f"Training completed, accuracy: {max(val_accuracy)}")
    model = model.float()
    return val_accuracy


def train_gpt(model: nn.Module, train_dataloader, val_dataloader, args) -> List[float]:
    assert torch.cuda.is_available(), "CUDA is not supported."
    torch_opt_on()
    model.cuda()

    def init_kernel_weights(m):
        if isinstance(m, Placeholder):
            if m.kernel and hasattr(m.kernel, "weights"):
                for w in m.kernel.weights:
                    nn.init.normal_(w, std=0.02)

    model.apply(init_kernel_weights)
    model.initialize_weights()
    optimizer = get_gpt_optimizer(model, args)

    num_iters = 0
    model.train()
    data_iterator = iter(train_dataloader)
    losses = []
    start_time = time.time()
    last_time = time.time()
    while True:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_dataloader)
            batch = next(data_iterator)

        batch = batch.cuda()
        _, loss = model(batch[:, :-1], batch[:, 1:])

        if time.time() - last_time > args.gpt_log_interval:
            last_time = time.time()
            value = loss.item()
            losses.append((time.time(), value))
            logging.info(f"Step: {num_iters}, train loss: {value}")

        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
        optimizer.step()

        if args.gpt_max_iters > 0 and num_iters >= args.gpt_max_iters:
            logging.info(f"Reaching max iterations {args.gpt_max_iters}, break")
            break

        if (
            args.gpt_max_minutes > 0
            and (time.time() - start_time) / 60 > args.gpt_max_minutes
        ):
            logging.info(f"Reaching max time limit {args.gpt_max_minutes} mins, break")
            break

        # Pruning
        if loss.item() < 3:
            logging.info(f"Illegal kernel, skip")
            losses.append((time.time(), 2.99))
            break

        if time.time() - start_time > 60 and loss.item() > args.gpt_max_loss:
            logging.info(f"Prune loss (last item): {loss.item()}")
            break
        num_iters += 1

    return losses


def train_gnn(model: nn.Module, train_dataloader, val_dataloader, args) -> List[float]:
    assert torch.cuda.is_available(), "CUDA is not supported."
    torch_opt_on()
    model.cuda()

    model.apply(init_weights)
    loss_func = get_gnn_loss_func(args)
    optimizer = get_gnn_optimizer(model, args)
    data = train_dataloader

    accuracy = []
    for epoch in range(args.epochs):
        # Forward
        model.train()
        out = model(data)
        loss = loss_func(out[data.train_mask], data.y[data.train_mask])

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=args.grad_norm_clip, norm_type=2
        )
        optimizer.step()
        optimizer.zero_grad()

        # Validation
        model.eval()
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        accuracy.append(int(correct) / int(data.test_mask.sum()))

    logging.info(f"Training completed, accuracy: {max(accuracy)}")
    return accuracy
