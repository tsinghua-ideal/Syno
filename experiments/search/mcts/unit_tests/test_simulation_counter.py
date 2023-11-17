import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from simulation_counter import SimulationCounter

counter = SimulationCounter(6, increase_ratio=2)
trial_gen = counter.get_trials()
for i, trials in zip(range(10), trial_gen):
    assert trials == 2**i, f"{i}th trial is {trials}, not {2 ** i}"

counter.list_update([5 for _ in range(10)], [10 for _ in range(10)])
assert counter.estimated_probability() == 0.5
trial_gen = counter.get_trials()
for trial in trial_gen:
    assert trial == 2, trial
    break

counter.list_update([15 for _ in range(3)], [10 for _ in range(3)])
assert counter.estimated_probability() == 1.0
trial_gen = counter.get_trials()
for trial in trial_gen:
    assert trial == 1, trial
    break

print("[PASSED] test_simulation_counter")
