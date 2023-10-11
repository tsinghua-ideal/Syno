import logging
import torch
import time
from torch import nn
from typing import Tuple, List
from timm.utils import AverageMeter
from transformers import GPT2Tokenizer

from .loss import get_loss_func
from .optim import get_optimizer, get_gpt_optimizer
from .sche import get_schedule
from .models import KASModel


def torch_opt_on():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train(model, train_dataloader, val_dataloader, args, init_weight=True) -> List[float]:
    if 'gpt' in args.model:
        return train_gpt(model, train_dataloader, val_dataloader, args)

    assert isinstance(model, KASModel)

    torch_opt_on()
    assert torch.cuda.is_available(), "CUDA is not supported."
    model.cuda()
    if init_weight:
        model.initialize_weights()

    # Loss, optimizer and scheduler
    loss_func = get_loss_func(args)
    optimizer = get_optimizer(model, args)
    scheduler, sched_epochs = get_schedule(optimizer, args)

    val_accuracy = []
    num_updates = 0
    for epoch in range(sched_epochs):
        # Train
        start_time = time.time()
        model.train()
        loss_meter = AverageMeter()
        for i, (image, label) in enumerate(train_dataloader):
            # Forward
            image, label = image.cuda(), label.cuda()
            logits = model(image)
            loss = loss_func(logits, label)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.grad_norm_clip, norm_type=2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Update loss and scheduler
            num_updates += 1
            loss_meter.update(loss.item(), image.size(0))
            scheduler.step_update(num_updates=num_updates)
        scheduler.step(epoch=epoch + 1)
        elapsed_train_time = time.time() - start_time

        # Valiation
        start_time = time.time()
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
                    image = torch.cat([image, torch.zeros(shape, dtype=image.dtype).cuda()], dim=0)
                    shape = (args.batch_size - label.size(0), *label.size()[1:])
                    label = torch.cat([label, torch.zeros(shape, dtype=label.dtype).cuda() - 1], dim=0)

                # Inference
                logits = model(image)

                # Statistic
                pred = torch.argmax(logits, 1)
                correct += torch.sum(pred == label).item()
            val_accuracy.append(correct / total)
        elapsed_valid_time = time.time() - start_time
        logging.info(f'Epoch [{epoch + 1}/{sched_epochs}], train loss: {loss_meter.avg}, test accuracy: {correct / total}, training time: {elapsed_train_time}, validation time: {elapsed_valid_time}')
        if epoch > 0 and args.kas_inference_time_limit and elapsed_train_time > args.kas_inference_time_limit:
            logging.info(f'Inference time limit reached ({elapsed_train_time}s currently), stopping training ...')
            break

        # Temporary hack
        # TODO: make a pruning file
        if epoch == 9 and max(val_accuracy) < 0.3:
            logging.info(f'Accuracy too low, pruning ...')
            break

    logging.info(f'Training completed, accuracy: {max(val_accuracy)}')
    return val_accuracy


def train_gpt(model: nn.Module, train_dataloader, val_dataloader, args) -> List[float]:
    assert torch.cuda.is_available(), "CUDA is not supported."
    torch_opt_on()
    model.cuda()

    model.initialize_weights()
    optimizer = get_gpt_optimizer(model, args)

    num_iters = 0
    model.train()
    data_iterator = iter(train_dataloader)
    while True:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_dataloader)
            batch = next(data_iterator)

        batch = batch.cuda()
        _, loss = model(batch, batch)
        logging.info(f'Train loss: {loss.item()}')
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
        optimizer.step()

        if args.gpt_max_iters > 0 and num_iters >= args.gpt_max_iters:
            break
