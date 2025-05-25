import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="KAS gpt loss plot")
parser.add_argument("--baseline-gpt-loss", type=str, required=True)
parser.add_argument("--syno-gpt-loss", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

fig, ax = plt.subplots(figsize=(7, 3.5))

losses = [args.baseline_gpt_loss, args.syno_gpt_loss]
labels = ['GPT-2', 'Syno']
colors = ['#FA6F6F', '#82B0D2']

for loss, label, color in zip(losses, labels, colors):
    
    with open(loss, 'r') as f:
        data = eval(f.read())
        
    losses, times = zip(*data)
    cut = 1000
    losses = losses[cut: ]
    times = times[cut: ]
    times = [t - times[0] for t in times]

    window_size = 50
    window = np.ones(window_size) / window_size
    
    smoothed_losses = np.convolve(losses, window, 'valid')
    print(smoothed_losses.shape)
    smoothed_losses = np.exp(smoothed_losses)
    smoothed_times = times[:len(smoothed_losses)]
    smoothed_times = np.arange(len(smoothed_times))
    step = max(1, len(smoothed_losses) // 3000)
    max_steps = 100000
    smoothed_losses = smoothed_losses[:max_steps:step]
    smoothed_times = smoothed_times[:max_steps:step]

    ax.plot(smoothed_times, smoothed_losses, label=label, c=color, linewidth=1.3, linestyle="-")

ax.set_xlabel('Steps', fontsize=14)
ax.set_ylabel('Perplexity', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(axis='y', linestyle='--')

ax.legend(fontsize=14)

plt.savefig(f"{args.output}/gpt-loss.pdf", dpi=300, bbox_inches='tight')
plt.close()
