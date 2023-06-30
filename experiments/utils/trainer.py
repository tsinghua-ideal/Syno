import torch
import time
import random
from typing import Tuple, List
from timm.utils import AverageMeter

from .loss import get_loss_func
from .optim import get_optimizer
from .sche import get_schedule


def train(model, train_loader, val_loader, args) -> Tuple[List[float], List[float]]:
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available(), 'CUDA is not supported.'
    model.cuda()

    # Loss, optimizer and scheduler
    loss_func = get_loss_func(args)
    optimizer = get_optimizer(model, args)
    scheduler, sched_epochs = get_schedule(optimizer, args)

    # Prefetch all data into GPU
    if args.fetch_all_to_gpu:
        print('Fetching all data into GPU ...')
        train_loader = [(image, label) for (image, label) in train_loader]
        val_loader = [(image, label) for (image, label) in val_loader]

    train_errors = []
    val_errors = []
    start = time.time()
    num_updates = 0
    for epoch in range(sched_epochs):
        # Train
        start_time = time.time()
        model.train()
        loss_meter = AverageMeter()
        if args.fetch_all_to_gpu:
            random.shuffle(train_loader)
        for i, (image, label) in enumerate(train_loader):
            # Forward
            logits = model(image)
            loss = loss_func(logits, label)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            for i, (image, label) in enumerate(val_loader):
                # Inference
                logits = model(image)

                # Statistic
                pred = torch.argmax(logits, 1)
                correct += torch.sum(pred == label).item()
                total += label.size(0)
            val_errors.append(1 - correct / total)
        elapsed_valid_time = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{sched_epochs}], train loss: {loss_meter.avg}, test accuracy: {correct / total}, training time: {elapsed_train_time}, validation time: {elapsed_valid_time}')

    print(f'Training completed, accuracy: {1 - min(val_errors)}')
    return train_errors, val_errors
