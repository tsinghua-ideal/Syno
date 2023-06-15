def get_batch(dataloader):
    images, labels = next(iter(dataloader))
    return images, labels