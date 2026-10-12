"""Prevent policy RNG consumption from changing environment randomization."""
from contextlib import contextmanager
import random
import numpy as np


class EnvironmentRNG:
    def __init__(self, seed):
        self.numpy = np.random.RandomState(seed).get_state()
        self.python = random.Random(seed).getstate()
        self.depth = 0

    @contextmanager
    def use(self):
        if self.depth:
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1
            return
        external = (np.random.get_state(), random.getstate())
        np.random.set_state(self.numpy)
        random.setstate(self.python)
        self.depth = 1
        try:
            yield
        finally:
            self.numpy, self.python = np.random.get_state(), random.getstate()
            np.random.set_state(external[0])
            random.setstate(external[1])
            self.depth = 0
