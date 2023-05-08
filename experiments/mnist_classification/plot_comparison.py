import matplotlib.pyplot as plt
import os
import json
import argparse
import numpy as np

def parse_perf(result_folder: str):
    
    assert os.path.exists(result_folder), f"{result_folder} does not exists!"
    
    perf_dict = json.load(open(os.path.join(result_folder, 'perf.json')))
    rewards = perf_dict['rewards']
    times = perf_dict['times']
    return times, rewards

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='KAS MNIST visualizer')

    # Dataset.
    parser.add_argument('--result1', type=str, default='./results',
                        help='Result folder of the first result')
    parser.add_argument('--task1', type=str, default='MCTS',
                        help='Result folder of the first result')
    parser.add_argument('--result2', type=str, default='./results_random',
                        help='Result folder of the second result')
    parser.add_argument('--task2', type=str, default='Random',
                        help='Result folder of the second result')
    parser.add_argument('--out_folder', type=str, default='./visualization',
                        help='Output folder')
    
    args = parser.parse_args()
    
    times1, rewards1 = parse_perf(args.result1)
    times2, rewards2 = parse_perf(args.result2)
    plt.figure()
    plt.subplot(221)
    plt.scatter(times1, rewards1, label=args.task1, marker='o')
    plt.scatter(times2, rewards2, label=args.task2, marker='^')
    plt.xlabel("Times (unit: sec)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy w.r.t. time")
    plt.legend()
    
    avgreward1 = np.cumsum(rewards1) / (np.arange(len(rewards1)) + 1)
    avgreward2 = np.cumsum(rewards2) / (np.arange(len(rewards2)) + 1)
    plt.subplot(222)
    plt.plot(times1, avgreward1, label=args.task1, marker='o')
    plt.plot(times2, avgreward2, label=args.task2, marker='^')
    plt.xlabel("Times (unit: sec)")
    plt.ylabel("Accuracy")
    plt.title("Average Accuracy w.r.t. time")
    plt.legend()
    
    maxreward1 = [np.max(rewards1[:i+1]) for i in range(len(rewards1))]
    maxreward2 = [np.max(rewards2[:i+1]) for i in range(len(rewards2))]
    plt.subplot(223)
    plt.plot(times1, maxreward1, label=args.task1, marker='o')
    plt.plot(times2, maxreward2, label=args.task2, marker='^')
    plt.xlabel("Times (unit: sec)")
    plt.ylabel("Accuracy")
    plt.title("Maximum Accuracy w.r.t. time")
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(args.out_folder, exist_ok=True)
    plt.savefig(os.path.join(args.out_folder, "comparison.jpg"))
    
    