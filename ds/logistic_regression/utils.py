import numpy as np
from nptyping import NDArray, Number


def sigmoid(x: NDArray) -> NDArray:
    """Sigmoid function"""

    return 1 / (1 + np.exp(-x))


def logistic_cost_function(x: NDArray, betha: NDArray, y: NDArray) -> Number:

    y_hat = sigmoid(x.dot(betha))

    costs = y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)

    return -costs.mean()


def logistic_cost_gradient(x: NDArray, betha: NDArray, y: NDArray) -> NDArray:
    """Return gradient vector of the logistic regression cost function"""

    if not x.shape[0] == y.shape[0]:
        raise ValueError(f"Wrong shape: x -> {x.shape} - y -> {y.shape}")

    return x.T.dot((y - sigmoid(x.dot(betha)).sum(axis=0)))


def logistic_cost_hessian(x: NDArray, betha: NDArray, y: NDArray) -> NDArray:

    if not x.shape[0] == y.shape[0]:
        raise ValueError(f"Wrong shape: x -> {x.shape} - y -> {y.shape}")

    y_hat = sigmoid(x.dot(betha))

    return x.T.dot(np.diag(y_hat * (1 - y_hat))).dot(x)
