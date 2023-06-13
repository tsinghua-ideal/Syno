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
    parser.add_argument('--result1', type=str, default='./final_result',
                        help='Result folder of the first result')
    parser.add_argument('--task1', type=str, default='2-step MCTS',
                        help='Result folder of the first result')
    # parser.add_argument('--result2', type=str, default='./results_random',
    #                     help='Result folder of the second result')
    # parser.add_argument('--task2', type=str, default='Random',
    #                     help='Result folder of the second result')
    parser.add_argument('--out_folder', type=str, default='./visualization',
                        help='Output folder')
    
    args = parser.parse_args()
    
    # results = [args.result1, args.result2, './results_1step']
    # tasks = [args.task1, args.task2, 'original MCTS']
    results = [args.result1]
    tasks = [args.task1]
    
    times_and_rewards = [parse_perf(res) for res in results]
    times = [tr[0] for tr in times_and_rewards]
    rewards = [tr[1] for tr in times_and_rewards]
    
    plt.figure()
    plt.subplot(221)
    for time, reward, task in zip(times, rewards, tasks):
        plt.scatter(time, reward, label=task, marker='o')
    plt.xlabel("Times (unit: sec)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy w.r.t. time")
    plt.legend()
    
    avgrewards = [np.cumsum(reward) / (np.arange(len(reward)) + 1) for reward in rewards]
    plt.subplot(222)
    for time, reward, task in zip(times, avgrewards, tasks):
        plt.plot(time, reward, label=task, marker='o')
    plt.xlabel("Times (unit: sec)")
    plt.ylabel("Accuracy")
    plt.title("Average Accuracy w.r.t. time")
    plt.legend()
    
    maxrewards = [[np.max(reward[:i+1]) for i in range(len(reward))] for reward in rewards]
    plt.subplot(223)
    for time, reward, task in zip(times, maxrewards, tasks):
        plt.plot(time, reward, label=task, marker='o')
    plt.xlabel("Times (unit: sec)")
    plt.ylabel("Accuracy")
    plt.title("Maximum Accuracy w.r.t. time")
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(args.out_folder, exist_ok=True)
    plt.savefig(os.path.join(args.out_folder, "comparison.jpg"))
    
    print("Best performance", maxrewards[0][-1])
    
    