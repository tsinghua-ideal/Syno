import os
import math
import numpy as np
import matplotlib.pyplot as plt

directory = './results/loss'

fig, ax = plt.subplots()

for filename in os.listdir(directory):
    if filename.endswith('.out') and ('linear' in filename or '179' in filename):
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'r') as f:
            data = eval(f.read())
            
        times, losses = zip(*data)
        times = [t - times[0] for t in times]
        times = [t for t in times if t <= 1000]
        losses = losses[:len(times)]
        losses = [math.exp(l) for l in losses]
        
        # losses = np.cumsum(losses) / np.arange(1, len(losses) + 1)
        ax.plot(times, losses, label=filename)

ax.set_xlabel('Time')
ax.set_ylabel('Average Perplexity')
ax.set_title('Average Perplexity Over Time')

ax.legend()

output_dir = "results/loss"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/plot.png", dpi=300)
plt.close()
