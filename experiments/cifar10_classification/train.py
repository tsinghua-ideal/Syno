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
    val_period=1,
    use_cuda=True,
    verbose=False
) -> Tuple[List[float], List[float]]:

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA is not supported. "
        model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    scheduler, sched_epochs = create_scheduler(args, optimizer)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)

    best_model_state_dict = {}

    train_errors = []
    val_errors = []

    scaler = torch.cuda.amp.GradScaler()

    start = time.time()
    for epoch in range(sched_epochs):
        correct = 0
        total = 0
        losses_m = AverageMeter()
        start = time.time()
        model.train()
        for i, (image, label) in enumerate(train_loader):
            image = image.cuda()
            label = label.cuda()

            # inference
            with torch.cuda.amp.autocast():
                logits = model(image)
                loss = criterion(logits, label)

            # backward
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=5., norm_type=2)
            scaler.step(optimizer)
            scaler.update()

            # statistic
            pred = torch.argmax(logits, 1)
            correct += torch.sum(pred == label).item()
            total += label.size(0)
            losses_m.update(loss.item(), image.size(0))

            scheduler.step_update(
                num_updates=epoch * len(train_loader) + i + 1, metric=losses_m.avg)

        train_errors.append(1 - correct / total)
        # scheduler.step()

        if (epoch + 1) % val_period == 0:
            with torch.no_grad():
                model.eval()
                correct = 0
                total = 0
                for i, (image, label) in enumerate(val_loader):

                    image = image.cuda()
                    label = label.cuda()

                    # inference
                    logits = model(image)

                    # statistic
                    pred = torch.argmax(logits, 1)
                    correct += torch.sum(pred == label).item()
                    total += label.size(0)

                val_errors.append(1 - correct / total)
                if val_errors[-1] == min(val_errors):
                    best_model_state_dict = model.state_dict()

            if verbose:
                logging.info(
                    f'Epoch {epoch+1}, train loss {losses_m.avg}, train error {train_errors[-1]}, validation error {val_errors[-1]}, elapsed {time.time() - start}')
                start = time.time()

        scheduler.step(epoch + 1, 1-val_errors[-1])

    print(f'Training Complete. Accuracy {1-val_errors[-1]}')

    return train_errors, val_errors, best_model_state_dict


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    args = arg_parse()
    train_data_loader, validation_data_loader = get_dataloader(args)

    model = ConvNet()

    train(model, train_data_loader, validation_data_loader,
          use_cuda=torch.cuda.is_available(), verbose=True)
