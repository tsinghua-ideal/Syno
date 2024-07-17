import multiprocessing as mp
import os
import time
from typing import Optional
import unittest
from unittest import mock

# go to parent directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grid_tune
from grid_tune import SerializableInstance

def mock_run_tuning(instance: SerializableInstance, progress: "mp.Queue[Optional[int]]"):
    time.sleep(0.1)
    progress.put(instance.trials)
    progress.put(None)

grid_tune.run_tuning = mock_run_tuning

class TestGridTune(unittest.TestCase):
    def test(self):
        grid_tune.main()

if __name__ == "__main__":
    unittest.main()
