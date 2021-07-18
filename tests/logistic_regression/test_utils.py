import numpy as np
import pytest

from ds.logistic_regression.utils import (
    logistic_cost_function,
    logistic_cost_gradient,
    logistic_cost_hessian,
    sigmoid,
)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])),
        (np.array([0, np.inf, -np.inf]), np.array([0.5, 1.0, 0.0])),
    ],
)
def test_sigmoid(test_input: float, expected: float):

    comparison = sigmoid(test_input) == expected
    if isinstance(comparison, np.ndarray):
        assert comparison.all()
    else:
        assert comparison


def test_logistic_cost_derivatives_raise_assertion_errors():

    with pytest.raises(ValueError):
        logistic_cost_gradient(np.array([1, 2]), np.array([2]), np.array([1]))

    with pytest.raises(ValueError):
        logistic_cost_hessian(np.array([1, 2]), np.array([2]), np.array([1]))


def test_logistic_cost_function():
    assert (
        logistic_cost_function(
            np.array([[1, 2], [2, 4], [4, 8]]),
            np.array([-2, 1]),
            np.array([0.5, 0.5, 0.5]),
        )
        == pytest.approx(0.6931471805599453)
    )


def test_logistic_cost_regression():

    assert (
        logistic_cost_gradient(
            np.array([[1, 2], [2, 4], [4, 8]]),
            np.array([-2, 1]),
            np.array([1.5, 1.5, 1.5]),
        )
        == np.array([0, 0])
    ).all()
