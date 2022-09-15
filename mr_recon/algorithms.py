import numpy as np
import numpy.typing as npt

from operators import LinearOperator
from norms import Norm


class PDHG:

    def __init__(self,
                 data: npt.NDArray,
                 data_operator: LinearOperator,
                 data_norm: Norm,
                 prior_operator: LinearOperator,
                 prior_norm: Norm,
                 beta: float,
                 sigma: float,
                 tau: float,
                 theta: float = 0.999) -> None:

        self._data = data

        self._data_operator = data_operator
        self._data_norm = data_norm

        self._prior_operator = prior_operator
        self._prior_norm = prior_norm

        self._beta = beta

        self._sigma = sigma
        self._tau = tau
        self._theta = theta

        self.initialize()

    @property
    def x(self) -> npt.NDArray:
        return self._x

    @property
    def y_data(self) -> npt.NDArray:
        return self._y_data

    @property
    def y_prior(self) -> npt.NDArray:
        return self._y_prior

    def initialize(self) -> None:
        self._x = np.zeros(self._data_operator.x_shape)
        self._xbar = np.zeros(self._data_operator.x_shape)

        self._y_data = np.zeros(self._data_operator.y_shape)
        self._y_prior = np.zeros(self._prior_operator.y_shape)

        self._iteration_number = 0
        self._cost = []

    def update(self) -> None:
        # data forward step
        self._y_data = self._y_data + (
            self._sigma *
            (self._data_operator.forward(self._xbar) - self._data))

        # prox of data fidelity
        self._y_data = self._data_norm.prox_convex_dual(self._y_data,
                                                        sigma=self._sigma)

        # prior operator forward step
        self._y_prior = self._y_prior + (
            self._sigma * self._prior_operator.forward(self._xbar))
        # prox of prior norm
        self._y_prior = self._beta * self._prior_norm.prox_convex_dual(
            self._y_prior / self._beta, sigma=self._sigma / self._beta)

        x_plus = self._x - self._tau * self._data_operator.adjoint(
            self._y_data) - self._tau * self._prior_operator.adjoint(
                self._y_prior)

        self._xbar = x_plus + self._theta * (x_plus - self._x)

        self._x = x_plus.copy()

        self._iteration_number += 1

    def run(self,
            num_iterations: int,
            calculate_cost: bool = False,
            verbose: bool = True) -> None:
        for i in range(num_iterations):
            self.update()
            if verbose:
                print(f'iteration {self._iteration_number}')
            if calculate_cost:
                self._cost.append(
                    self._data_norm(self._data_operator.forward(self._x)) +
                    self._beta *
                    self._prior_norm(self._prior_operator.forward(self._x)))
