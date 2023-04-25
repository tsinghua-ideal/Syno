import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import time
from typing import Tuple, List

from utils.models import KASConv
from utils.data import get_dataloader


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion=nn.CrossEntropyLoss(),
    lr=0.1,
    momentum=0.9,
    epochs=50,
    val_period=1,
    use_cuda=True,
    verbose=False
) -> Tuple[List[float], List[float]]:

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA is not supported. "
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.train()
    train_errors = []
    val_errors = []

    start = time.time()
    for epoch in range(epochs):
        correct = 0
        total = 0
        train_loss = 0
        for i, (image, label) in enumerate(train_loader):

            image = image.cuda()
            label = label.cuda()

            # inference
            logits = model(image)
            loss = criterion(logits, label)

            # statistic
            pred = torch.argmax(logits, -1)
            correct += torch.sum(pred == label).item()
            total += label.size(0)
            train_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        if (epoch + 1) % 5 == 0 and verbose:
            print(
                f'Epoch {epoch+1}, train loss {train_loss}, train error {train_errors[-1]}, validation error {val_errors[-1]}, elapsed {time.time() - start}')
            start = time.time()

    return train_errors, val_errors


if __name__ == '__main__':

    train_data_loader, validation_data_loader = get_dataloader()

    model = KASConv()

    train(model, train_data_loader, validation_data_loader,
          use_cuda=torch.cuda.is_available())
