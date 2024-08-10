import os
import math
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.signal import savgol_filter
window_length = 31 
polyorder = 2  

directory = './logs/manual/rwkv-v1'
output_dir = os.path.join(directory, "loss")
model_name = "rwkv/rwkv-v5.1a-0.1b"

baseline = ('baseline.log', 'Original Model')
data = [
     ('04917_579825011801549874.log', 'Kernel 1'), 
     ('04930_8620085500577985662.log', 'Kernel 2'), 
     ('04951_13182919585335735127.log', 'Kernel 3'), 
     ('04966_15475986206679792241.log', 'Kernel 4'),
     ('04981_15003962989689026618.log', 'Kernel 5'),
     ('05017_7032845535107897390.log', 'Kernel 6')
]

os.makedirs(output_dir, exist_ok=True)

def analyze_log(log_content):
    # Regular expression to match the relevant lines
    pattern = re.compile(r"INFO Step: (\d+), train loss: ([\d.]+)")
    pattern_flops_base = re.compile(r"INFO Base model rwkv/rwkv-v5.1a-0.1b has ([\d.]+) GFLOPs")
    pattern_flops_replaced = re.compile(r"INFO Loaded model has ([\d]+) FLOPs")
    base_flops = None
    replaced_flops = None

    # Extract steps and losses
    steps, losses = [], []
    for line in log_content:
        match = pattern.search(line.strip())
        match_flops_base = pattern_flops_base.search(line.strip())
        match_flops_replaced = pattern_flops_replaced.search(line.strip())
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            steps.append(step)
            losses.append(loss)
        if match_flops_base:
            base_flops = float(match_flops_base.group(1))
        if match_flops_replaced:
            print(match_flops_replaced.group(1))
            replaced_flops = float(match_flops_replaced.group(1)) / (2 ** 30)
    
    flops = replaced_flops if replaced_flops else base_flops
    return np.array(steps), np.array(losses), flops

for filename, label in data:
    baseline_fn, baseline_label = baseline
    baseline_fn = os.path.join(directory, baseline_fn)
    filepath = os.path.join(directory, filename)

    fig, ax = plt.subplots()
    steps_baseline, losses_baseline, _ = analyze_log(open(baseline_fn).readlines())
    steps, losses, _ = analyze_log(open(filepath).readlines())
    
    ax.plot(steps_baseline, savgol_filter(losses_baseline, window_length, polyorder), label=baseline_label, color='lightskyblue')
    ax.plot(steps, savgol_filter(losses, window_length, polyorder), label=label, color='coral')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Log (Perplexity)')
    ax.set_title('Average Perplexity Over Time')
    ax.legend()

    plt.savefig(f"{output_dir}/{label}_curve.pdf", dpi=300)
    plt.clf()

fig, ax = plt.subplots()
for filename, label in [baseline] + data:
        filepath = os.path.join(directory, filename)
            
        steps, losses, gflops = analyze_log(open(filepath).readlines())
        mean_loss = losses[steps >= 0.8 * steps.max()].mean()
        
        ax.scatter([gflops], [mean_loss], label=label)

ax.set_xlabel('GFLOPs')
ax.set_ylabel('Log (Perplexity)')
ax.set_ylim(bottom=4.44, top=4.54)
ax.set_xlim(left=100, right=260)
ax.set_title('Average Perplexity')
ax.legend()

plt.savefig(f"{output_dir}/plot-scatter.pdf", dpi=300)
