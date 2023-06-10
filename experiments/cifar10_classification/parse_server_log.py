"""
Parse the log of servers. Useful before the training ends. 
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

import argparse
import re
import os
from tqdm.contrib import tzip

from utils.parser import arg_parse
from mp_utils import Handler_client
from utils.models import KASConv, ModelBackup

from KAS import Sampler, TreePath
from KAS.Bindings import CodeGenOptions


def arg_parse_logger():
    parser = argparse.ArgumentParser(
        description='Log Parser')

    parser.add_argument('--log_path', type=str,
                        default='./logs/server/stdout.log', help='Path to the log to be parsed')
    parser.add_argument('--output_path', type=str,
                        default='./parsed', help='Path to the output')

    args = parser.parse_args()
    return args


def create_sample(args_logger, path):
    args = arg_parse()

    web_handler = Handler_client(args)
    sampler_args, train_args, extra_args = web_handler.get_args()

    sampler_args['autoscheduler'] = getattr(
        CodeGenOptions.AutoScheduler, sampler_args['autoscheduler'])
    sampler_args['save_path'] = os.path.join(
        args_logger.output_path, 'samples')

    _model = ModelBackup(KASConv, torch.randn(
        extra_args["sample_input_shape"]), extra_args["device"])
    kas_sampler = Sampler(net=_model.create_instance(), **sampler_args)
    node = kas_sampler.visit(path)
    kernelPacks, total_flops = kas_sampler.realize(
        _model.create_instance(), node, extra_args["prefix"])
    print(f"Total FLOPS: {total_flops}")


def main():
    args = arg_parse_logger()

    os.makedirs(args.output_path, exist_ok=True)
    assert os.path.exists(args.log_path)

    paths = []
    rewards = []
    summary = []
    summary_flag = False

    log_file = open(args.log_path, 'r').readlines()
    for i, line in enumerate(log_file):
        if i < len(log_file) - 1 and log_file[i+1] == "Successfully updated MCTS. \n":
            path_raw = re.search("name=[\d|_]+", line).group()[len("name="):]
            path = TreePath.deserialize(path_raw)
            reward = float(re.search("\$[\d|\.]+", line).group()[1:])
            paths.append(path)
            rewards.append(reward)
        elif line == "**************** Logged Summary ******************\n":
            summary_flag = not summary_flag
            if summary_flag:
                summary.append([])
        elif summary_flag:
            summary[-1].append(line)

    assert len(paths) == len(rewards) == len(
        summary), f"{len(paths)} {len(rewards)} {len(summary)}"
    perf = zip(rewards, paths)
    best_reward, best_path = max(perf, key=lambda x: x[0])

    # create_sample(args, best_path)

    avgrewards = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    plt.plot(avgrewards, marker='o')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Average Accuracy w.r.t. time")
    plt.savefig(os.path.join(args.output_path, 'AverageReward.jpg'))
    plt.clf()

    maxrewards = [np.max(rewards[:i+1]) for i in range(len(rewards))]
    plt.plot(maxrewards, marker='o')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Maximum Accuracy w.r.t. time")
    plt.savefig(os.path.join(args.output_path, 'MaximumReward.jpg'))

    with open(os.path.join(args.output_path, 'parsed.log'), 'w') as f_out:
        f_out.write(
            "****************** Performance Summary ******************\n")
        f_out.write(f"The best reward found: {best_reward}\n")
        f_out.write(f"The corresponding path: {best_path}\n")
        f_out.write(f"Statistics at the end: \n")
        f_out.writelines(summary[-1])
        f_out.write(
            "****************** Performance Summary ******************\n\n")

        for i, (path, reward, summ) in enumerate(tzip(paths, rewards, summary)):
            f_out.write(
                f"****************** Performance of Iteration {i+1} ******************\n")
            f_out.write(f"The reward: {reward}\n")
            f_out.write(f"The corresponding path: {path}\n")
            f_out.write(f"Statistics: \n")
            f_out.writelines(summ)
            f_out.write(f"\n")


if __name__ == '__main__':
    main()
