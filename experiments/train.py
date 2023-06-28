import torch
from torch import nn
from torch.utils.data import DataLoader

import time
import logging
from typing import Tuple, List
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import AverageMeter

from utils.models import ConvNet
from utils.data import get_dataloader
from utils.parser import arg_parse


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
) -> Tuple[List[float], List[float]]:
    # Use CUDA by default
    assert torch.cuda.is_available(), 'CUDA is not supported.'
    model.cuda()

    # Loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    scheduler, sched_epochs = create_scheduler(args, optimizer)

    best_model_state_dict = {}

    train_errors = []
    val_errors = []

    start = time.time()
    num_updates = 0
    for epoch in range(sched_epochs):
        # Train
        start_time = time.time()
        model.train()
        loss_meter = AverageMeter()
        for i, (image, label) in enumerate(train_loader):
            # Forward
            logits = model(image)
            loss = criterion(logits, label)

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
            if val_errors[-1] == min(val_errors):
                best_model_state_dict = model.state_dict()
        elapsed_valid_time = time.time() - start_time
        logging.info(f'Epoch [{epoch + 1}/{sched_epochs}], train loss: {loss_meter.avg}, test accuracy: {correct / total}, training time: {elapsed_train_time}, validation time: {elapsed_valid_time}')

    print(f'Training complete, accuracy: {1 - min(val_errors)}')
    return train_errors, val_errors, best_model_state_dict


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    args = arg_parse()
    train_data_loader, validation_data_loader = get_dataloader(args)

    model = ConvNet()

    train(model, train_data_loader, validation_data_loader,
          use_cuda=torch.cuda.is_available(), verbose=True)
