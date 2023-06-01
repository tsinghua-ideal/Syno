import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import time
import logging
from typing import Tuple, List
from tqdm import tqdm

from utils.models import ConvNet
from utils.data import get_dataloader
from utils.parser import arg_parse


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr,
    epochs,
    val_period=1,
    use_cuda=True,
    verbose=False
) -> Tuple[List[float], List[float]]:

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA is not supported. "
        model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_model_state_dict = {}

    model.train()
    train_errors = []
    val_errors = []

    scaler = torch.cuda.amp.GradScaler()

    start = time.time()
    for epoch in range(epochs):
        correct = 0
        total = 0
        train_loss = 0
        for i, (image, label) in enumerate(train_loader):

            image = image.cuda()
            label = label.cuda()

            # inference
            with torch.cuda.amp.autocast():
                logits = model(image)
                loss = F.cross_entropy(logits, label)

            # statistic
            pred = torch.argmax(logits, -1)
            correct += torch.sum(pred == label).item()
            total += label.size(0)
            train_loss += loss.item()

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

        train_errors.append(1 - correct / total)
        train_loss /= len(train_loader)

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
                    pred = torch.argmax(logits, -1)
                    correct += torch.sum(pred == label).item()
                    total += label.size(0)

                val_errors.append(1 - correct / total)
                if val_errors[-1] == min(val_errors):
                    best_model_state_dict = model.state_dict()

        if (epoch + 1) % val_period == 0 and verbose:
            logging.info(
                f'Epoch {epoch+1}, train loss {train_loss}, train error {train_errors[-1]}, validation error {val_errors[-1]}, elapsed {time.time() - start}')
            start = time.time()

    print(f'Training Complete. Accuracy {1-val_errors[-1]}')

    return train_errors, val_errors, best_model_state_dict


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    args = arg_parse()
    train_data_loader, validation_data_loader = get_dataloader(args)

    model = ConvNet()

    train(model, train_data_loader, validation_data_loader,
          use_cuda=torch.cuda.is_available(), verbose=True)
