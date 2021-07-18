from typing import Callable

import numpy as np


def newton(func: Callable, x0: np.ndarray, fprime: Callable):

    step = x0 / fprime(x0)
