from typing import Union

import numpy as np


@np.vectorize
def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Sigmoid function"""

    if x == np.inf:
        return 1.0
    elif x == -np.inf:
        return 0.0

    return 1 / (1 + np.exp(-x))


def logistic_cost_function(x: np.ndarray, betha: np.ndarray, y: np.ndarray) -> float:

    y_hat = sigmoid(x.dot(betha))

    costs = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)

    return -costs.mean()


def logistic_cost_gradient(
    x: np.ndarray, betha: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Return gradient vector of the logistic regression cost function"""

    assert x.shape[0] == y.shape[0]

    return x.T.dot((y - sigmoid(x.dot(betha)).sum(axis=0)))
