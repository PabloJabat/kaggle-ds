from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls


def get_age_bucket(age: int) -> str:
    step = 25
    for limit in range(0, 100 + step, step):
        if age < limit:
            return f"{int(limit - step)}_TO_{int(limit - 1)}"

    raise Exception


def value_counts(array: np.array) -> Dict[Any, int]:
    array_counts = {}
    for a_i in array:
        if a_i not in array_counts:
            array_counts[a_i] = 0
        array_counts[a_i] += 1
    return array_counts


def fit_model(x: np.array, y: np.array) -> Tuple[np.array, float]:
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    return nnls(x, y)


def show_hist(data: np.array) -> None:
    # data is unidimensional
    assert len(data.shape) == 1

    plt.hist(data)
    plt.show()


def encode_data(data: np.array) -> np.array:

    assert len(data.shape) == 1

    mapping = {key: i for i, key in enumerate(value_counts(data).keys())}

    return (
        np.array([mapping[key] for key in data], dtype=np.int64).reshape(-1, 1),
        mapping,
    )


def _sum_of_squares(a: np.array, b: Union[np.array, float]) -> float:
    assert len(a.shape) == 1
    assert len(b.shape) == 1 or isinstance(b, float)

    return np.sqrt(sum((a - b) * (a - b)))


def total_sum_of_squares(data: np.array) -> float:

    return _sum_of_squares(data, data.mean())


def residual_sum_of_squares(data: np.array, data_p: np.array) -> float:

    return _sum_of_squares(data, data_p)


def r_squared(data: np.array, data_p: np.array) -> float:

    return 1 - residual_sum_of_squares(data, data_p) / total_sum_of_squares(data)
