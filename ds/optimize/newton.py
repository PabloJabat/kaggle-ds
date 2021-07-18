# usr/bin/env python3
"""Newton optimization"""

from typing import Any, Callable

import numpy as np
from nptyping import NDArray


def newton_optimization(
    func: Callable[[Any], NDArray],
    x0: NDArray,
    fprime: Callable[[Any], NDArray],
    iterations: int,
):
    """Perform the newton optimization"""

    current_x = x0

    for _ in range(iterations):
        step = func(current_x).dot(np.linalg.inv(fprime(current_x)))
        current_x -= step

    return current_x
