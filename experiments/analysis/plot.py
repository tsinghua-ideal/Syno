import argparse
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    # Get Path
    parser = argparse.ArgumentParser(description='KAS session plot')
    parser.add_argument('--dirs', type=str, nargs='+', default=None)
    parser.add_argument('--output', type=str, default='plot')
    parser.add_argument('--time', default=False, action='store_true')
    args = parser.parse_args()
    assert args.dirs is not None
    for dir in args.dirs:
        assert os.path.exists(dir) and os.path.isdir(dir)

    # Read
    all_kernels = []
    for dir in args.dirs:
        kernels = []
        for kernel_fmt in os.listdir(dir):
            kernel_dir = os.path.join(dir, kernel_fmt)
            if not os.path.isdir(kernel_dir):
                continue
            if 'ERROR' in kernel_dir:
                continue
            files = list(os.listdir(kernel_dir))
            assert 'graph.dot' in files and 'loop.txt' in files and 'meta.json' in files

            meta_path = os.path.join(kernel_dir, 'meta.json')
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            kernels.append((meta['time'], meta['accuracy']))
        kernels = sorted(kernels, key=lambda x: x[0])
        if not args.time:
            kernels = list([(i, kernels[i][1]) for i in range(len(kernels))])
        all_kernels.append((dir, kernels))

    # Trend figure
    plt.figure(1, figsize=(25, 6), dpi=300)
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for name, kernels in all_kernels:
        x, y = zip(*kernels)
        y_sum, y_avg = 0, []
        for i in range(len(y)):
            y_sum += y[i]
            y_avg.append(y_sum / (i + 1))
        m, c = markers.pop(0), colors.pop(0)
        markers.append(m)
        colors.append(c)
        plt.plot(x, y_avg, marker=m, color=c, label=name, markersize=3)

    # Plot and save into file
    plt.xlabel('Time' if args.time else 'Samples')
    plt.ylabel('Accuracy (avg)')
    plt.legend()
    plt.savefig(f'{args.output}-avg.png')

    # Max figure
    plt.figure(2, figsize=(25, 6), dpi=300)
    for name, kernels in all_kernels:
        x, y = zip(*kernels)
        y_max = []
        for i in range(len(y)):
            y_max.append(y[i] if i == 0 else max(y_max[-1], y[i]))
        m, c = markers.pop(0), colors.pop(0)
        markers.append(m)
        colors.append(c)
        plt.plot(x, y_max, marker=m, color=c, label=name, markersize=3)

    # Plot and save into file
    plt.xlabel('Time' if args.time else 'Samples')
    plt.ylabel('Accuracy (max)')
    plt.legend()
    plt.savefig(f'{args.output}-max.png')

    # Histogram figure
    plt.figure(3, figsize=(10, 6), dpi=300)
    for name, kernels in all_kernels:
        x, y = zip(*kernels)
        m, c = markers.pop(0), colors.pop(0)
        markers.append(m)
        colors.append(c)
        sns.kdeplot(y, color=c, label=name, fill=True, bw_adjust=0.2, cut=0)
    
    # Plot and save into file
    plt.xlabel('Accuracy')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'{args.output}-hist.png')