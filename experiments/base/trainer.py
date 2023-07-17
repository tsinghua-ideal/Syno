import logging
import torch
import time
import random
from typing import Tuple, List
from timm.utils import AverageMeter

from .loss import get_loss_func
from .optim import get_optimizer
from .sche import get_schedule
from .models import KASModel


def train(model, train_dataloader, val_dataloader, args) -> Tuple[List[float], List[float]]:
    assert isinstance(model, KASModel)
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available(), 'CUDA is not supported.'
    model.cuda()
    model.initialize_weights()

    # Loss, optimizer and scheduler
    loss_func = get_loss_func(args)
    optimizer = get_optimizer(model, args)
    scheduler, sched_epochs = get_schedule(optimizer, args)

    train_errors = []
    val_errors = []
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
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Update loss and scheduler
            num_updates += 1
            loss_meter.update(loss.item(), image.size(0))
            scheduler.step_update(num_updates=num_updates, metric=loss_meter.avg)
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
            val_errors.append(1 - correct / total)
        elapsed_valid_time = time.time() - start_time
        logging.info(f'Epoch [{epoch + 1}/{sched_epochs}], train loss: {loss_meter.avg}, test accuracy: {correct / total}, training time: {elapsed_train_time}, validation time: {elapsed_valid_time}')
        if args.kas_inference_time_limit and elapsed_train_time > args.kas_inference_time_limit:
            logging.info(f'Inference time limit reached, stopping training ...')
            break

    logging.info(f'Training completed, accuracy: {1 - min(val_errors)}')
    return train_errors, val_errors
