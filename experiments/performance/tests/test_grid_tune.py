import multiprocessing as mp
import os
import time
from typing import Callable, Optional
import unittest

# go to parent directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import MetaScheduleTuner
from MetaScheduleTuner import TuningConfig

def mock_tune_e2e(config: TuningConfig, on_eval: Optional[Callable[[], None]] = None):
    time.sleep(0.1)
    if on_eval is None:
        return
    for _ in range(config.num_trials):
        on_eval()

MetaScheduleTuner.tune_e2e = mock_tune_e2e

import grid_tune

# reduce the number of trials to speed up the test
grid_tune.GRIDS = grid_tune.GRIDS[-2:]

class TestGridTune(unittest.TestCase):
    def test_success(self):
        args = grid_tune._parse_args()
        grid_tune.main(args)
    def test_timeout(self):
        args = grid_tune._parse_args()
        args.timeout = 0.01
        grid_tune.main(args)

if __name__ == "__main__":
    unittest.main()
