from dataclasses import dataclass
import itertools
import matplotlib.pyplot as plt
from typing import List

@dataclass
class MeasurementRecord:
    latency: float
    accuracy: float
    name: str

def plot_performance_vs_accuracy(baseline: MeasurementRecord, samples: List[MeasurementRecord]):
    fig, ax = plt.subplots()
    ax.set_xlabel('Relative Performance')
    ax.set_ylabel('Error Rate (%)')
    plt.scatter
    ax.scatter(
        [baseline.latency / baseline.latency],
        [100 * (1.0 - baseline.accuracy)],
        c='yellow', label=f'Baseline ({baseline.name})'
    )
    ax.scatter(
        [baseline.latency / sample.latency for sample in samples],
        [100 * (1.0 - sample.accuracy) for sample in samples],
        c='blue', label='Samples'
    )
    sorted_samples = sorted(itertools.chain(samples, [baseline]), key=lambda sample: sample.accuracy, reverse=True)
    running_min = [sorted_samples[0]]
    running_min_latency = sorted_samples[0].latency
    for sample in sorted_samples[1:]:
        if sample.latency < running_min_latency:
            running_min.append(sample)
            running_min_latency = sample.latency
    ax.step(
        [baseline.latency / sample.latency for sample in running_min],
        [100 * (1.0 - sample.accuracy) for sample in running_min],
        c='lavender'
    )

    ax.legend()
    plt.show()

if __name__ == "__main__":
    plot_performance_vs_accuracy(
        MeasurementRecord(35.7305, 0.76953125, 'conv2d'),
        [
            MeasurementRecord(8.9591, 0.750719572368421, 'conv1d_shift1d'),
            # MeasurementRecord(31.8738, ?, 'conv2d_dilation'),
            MeasurementRecord(8.9335, 0.720703125, 'conv2d_group'),
            MeasurementRecord(7.04, 0.7393092105263158, 'conv2d_pool'), # Warning! this is not real latency, because of runtime error!
        ]
    )
