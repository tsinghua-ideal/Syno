import os
import math
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

directory = './results/gpt_losses'

fig, ax = plt.subplots(figsize=(7, 3.5))

for filename in ('baseline.log', '5.log'):
    filepath = os.path.join(directory, filename)
    
    with open(filepath, 'r') as f:
        data = eval(f.read())
        
    losses, times = zip(*data)
    # print(mean(np.exp(losses)[int(0.9 * len(losses)):]))
    cut = 1000
    losses = losses[cut: ]
    times = times[cut: ]
    times = [t - times[0] for t in times]

    window_size = 50
    window = np.ones(window_size) / window_size
    
    smoothed_losses = np.convolve(losses, window, 'valid')
    smoothed_losses = np.exp(smoothed_losses)
    smoothed_times = times[:len(smoothed_losses)]
    smoothed_times = np.arange(len(smoothed_times))
    step = max(1, len(smoothed_losses) // 3000)
    max_steps = 100000
    smoothed_losses = smoothed_losses[:max_steps:step]
    smoothed_times = smoothed_times[:max_steps:step]

    ax.plot(smoothed_times, smoothed_losses,
            label='GPT-2' if 'b' in filename else 'Syno',
            c='#FA6F6F' if 'b' in filename else '#82B0D2',
            linewidth=1.3, linestyle="-")

ax.set_xlabel('Steps', fontsize=14)
ax.set_ylabel('Perplexity', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(axis='y', linestyle='--')

ax.legend(fontsize=14)

output_dir = "./analysis/results"
plt.savefig(f"{output_dir}/gpt-loss.pdf", dpi=300, bbox_inches='tight')
plt.close()
