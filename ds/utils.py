from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from nptyping import NDArray, Number
from scipy.optimize import nnls


def get_age_bucket(age: int) -> str:
    step = 25
    for limit in range(0, 100 + step, step):
        if age < limit:
            return f"{int(limit - step)}_TO_{int(limit - 1)}"

    raise Exception


def value_counts(array: np.ndarray) -> Dict[Any, int]:
    array_counts = {}
    for a_i in array:
        if a_i not in array_counts:
            array_counts[a_i] = 0
        array_counts[a_i] += 1
    return array_counts


def fit_model(x: NDArray, y: NDArray) -> NDArray:
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    return np.array(nnls(x, y)[0])


def show_hist(data: NDArray) -> None:
    # data is unidimensional
    assert len(data.shape) == 1

    plt.hist(data)
    plt.show()


def encode_data(data: NDArray) -> NDArray:

    assert len(data.shape) == 1

    mapping = {key: i for i, key in enumerate(value_counts(data).keys())}

    return (
        np.array([mapping[key] for key in data], dtype=np.int64).reshape(-1, 1),
        mapping,
    )


def _sum_of_squares(a: NDArray, b: NDArray) -> Number:
    assert len(a.shape) == 1
    assert len(b.shape) == 1 or isinstance(b, float)

    return np.sqrt(sum((a - b) * (a - b)))


def total_sum_of_squares(data: NDArray) -> Number:

    return _sum_of_squares(data, data.mean())


def residual_sum_of_squares(data: NDArray, data_p: NDArray) -> Number:

    return _sum_of_squares(data, data_p)


def r_squared(data: NDArray, data_p: NDArray) -> Number:

    return 1 - residual_sum_of_squares(data, data_p) / total_sum_of_squares(data)
