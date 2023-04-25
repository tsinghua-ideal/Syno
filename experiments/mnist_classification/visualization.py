import json
import os

from matplotlib import pyplot as plt

def visualize(train_errors, val_errors, label):
    out_folder = os.path.join('vis', label)
    os.makedirs(out_folder, exist_ok=True)
    epochs = len(train_errors)
    plt.figure()
    plt.plot(list(range(1, epochs+1)), train_errors, label='train')
    plt.plot(list(range(1, epochs+1)), val_errors, label='validation')
    plt.legend(loc='best')
    plt.savefig(os.path.join(out_folder, 'errors.jpg'))
    with open(os.path.join(out_folder, 'errors.json'), 'w') as f:
        json.dump({'train': train_errors, 'val': val_errors}, f)
    plt.clf()
