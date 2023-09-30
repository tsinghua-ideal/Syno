import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from Tuner import Tuner


def test_operator():
    tuner = Tuner()
    operator_tuner = tuner.get_operator_tuner("./kernel_example.py")
    operator_tuner.tune()
    print("Final latency: ", operator_tuner.evaluate())

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_operator()
