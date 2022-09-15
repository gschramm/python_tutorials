import numpy as np
import numpy.typing as npt

from operators import LinearOperator, complex_view_of_real_array
from functionals import Norm


class PDHG:
    """generic primal-dual hybrid gradient algorithm (Chambolle-Pock) for optimizing
       data_norm(data_operator x) + beta*(prior_norm(prior_operator x))"""

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
        """
        Parameters
        ----------
        data : npt.NDArray
            array containing that data
        data_operator : LinearOperator
            operator mapping current image to expected data
        data_norm : Norm
            norm applied to (expected data - data)
        prior_operator : LinearOperator
            prior operator
        prior_norm : Norm
            prior norm
        beta : float
            weight of prior
        sigma : float
            primal step size 
        tau : float
            dual step size 
        theta : float, optional
            theta parameter, by default 0.999
        """

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

    @property
    def cost_data(self) -> npt.NDArray:
        return np.array(self._cost_data)

    @property
    def cost_prior(self) -> npt.NDArray:
        return np.array(self._cost_prior)

    @property
    def cost(self) -> npt.NDArray:
        return self.cost_data + self.cost_prior

    def initialize(self) -> None:
        self._x = np.zeros(self._data_operator.x_shape)
        self._xbar = np.zeros(self._data_operator.x_shape)

        self._y_data = np.zeros(self._data_operator.y_shape)
        self._y_prior = np.zeros(self._prior_operator.y_shape)

        self._iteration_number = 0
        self._cost_data = []
        self._cost_prior = []

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
                self._cost_data.append(
                    self._data_norm(
                        self._data_operator.forward(self._x) - self._data))
                self._cost_prior.append(
                    self._beta *
                    self._prior_norm(self._prior_operator.forward(self._x)))


def sum_of_squares_reconstruction(data: npt.NDArray) -> npt.NDArray:

    data_complex = complex_view_of_real_array(data)
    recon_complex = np.zeros_like(data_complex)

    for i in range(data.shape[0]):
        recon_complex[i, ...] = np.fft.ifftn(data_complex[i, ...],
                                             norm='ortho')

    recon = np.sqrt((np.abs(recon_complex)**2).sum(axis=0))

    return recon