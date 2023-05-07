import matplotlib.pyplot as plt
import os
import json
import argparse

def parse_perf(result_folder: str):
    
    assert os.path.exists(result_folder), f"{result_folder} does not exists!"
    
    perf_dict = json.load(open(os.path.join(result_folder, 'perf.json')))
    upper_bounds = perf_dict['upper_bounds']
    times = perf_dict['times']
    return times, upper_bounds

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
    
    plt.figure()
    plt.plot(*parse_perf(args.result1), label=args.task1)
    plt.plot(*parse_perf(args.result2), label=args.task2)
    plt.xlabel("Times (unit: sec)")
    plt.ylabel("Highest Accuracy")
    plt.legend()
    
    os.makedirs(args.out_folder, exist_ok=True)
    plt.savefig(os.path.join(args.out_folder, "comparison.jpg"))
    
    