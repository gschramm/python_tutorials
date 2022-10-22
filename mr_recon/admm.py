"""demo script to understand ADMM on a small scale example

   here we want to solve the problem
   argmin_x f(x) + g(Kx)

   with f(x) = 0.5||x - y||_2^2
        g(.) = ||.||_1
   and  K a random maxtrix operator
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from scipy.optimize import fmin_cg, minimize


class LinearOperator:

    def __init__(self, A: npt.NDArray) -> None:
        self._A = A

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._A.shape[1]

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._A.shape[0]

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        return self._A @ x

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        return self._A.T @ y


class SmoothFunction:

    def __init__(self, y: npt.NDArray) -> None:
        self._y = y

    def __call__(self, x: npt.NDArray) -> float:
        return 0.5 * np.sum((x - self._y)**2)

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        return x - self._y


class QuadMapFunction:

    def __init__(self, smoothFunction: SmoothFunction, rho: float,
                 linOperator: LinearOperator):
        self._smoothFunction = smoothFunction
        self._rho = rho
        self._linOperator = linOperator

    def __call__(self, x: npt.NDArray, v: npt.NDArray) -> float:
        return self._smoothFunction(x) + 0.5 * self._rho * (
            (self._linOperator(x) - v)**2).sum()

    def gradient(self, x: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
        return self._smoothFunction.gradient(
            x
        ) + self._rho * self._linOperator.adjoint(self._linOperator(x) - v)


class L1Norm:

    def __init__(self) -> None:
        pass

    def __call__(self, x: npt.NDArray) -> float:
        return np.abs(x).sum()

    def prox(self, x: npt.NDArray, lam: float) -> npt.NDArray:
        return np.clip(x - lam, 0, None) - np.clip(-x - lam, 0, None)


if __name__ == '__main__':
    np.random.seed(1)
    n = 10
    y = 50 * np.random.rand(n)
    func = SmoothFunction(y)

    A = np.zeros((n - 1, n))
    for i in range(n - 1):
        A[i, i] = -1
        A[i, i + 1] = 1

    linOp = LinearOperator(A)

    prior = L1Norm()

    niter = 30

    # initialize variables
    x = np.zeros(linOp.in_shape)
    z = np.zeros(linOp.out_shape)
    u = np.zeros(linOp.out_shape)

    # calculate reference solution via other optimizer
    total_func = lambda x: func(x) + prior(linOp(x))
    ref = minimize(total_func, x)

    rhos = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    cost = np.zeros((len(rhos), niter))
    x_admm = np.zeros((len(rhos), n))

    for irho, rho in enumerate(rhos):
        for it in range(niter):
            # solve ADMM subproblem (1) to update x
            func_quad = QuadMapFunction(func, rho, linOp)
            f = lambda x: func_quad(x, z - u)
            g = lambda x: func_quad.gradient(x, z - u)

            x0 = x.copy()
            x = fmin_cg(f, x0, g)

            # solve ADMM subproblem (2) to update z
            z = prior.prox(linOp(x) + u, 1. / rho)

            # update the dual variable
            u = u + (linOp(x) - z)

            cost[irho, it] = total_func(x)

        x_admm[irho, ...] = x.copy()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    for irho, rho in enumerate(rhos):
        ax[0].plot(np.arange(1, niter + 1),
                   cost[irho, ...],
                   label=f'rho = {rho}')
        ax[1].loglog(np.arange(1, niter + 1),
                     cost[irho, ...],
                     label=f'rho = {rho}')
    ax[0].legend()
    ax[0].set_ylim(0.999 * ref.fun, 1.05 * ref.fun)
    ax[0].axhline(ref.fun, lw=0.5, color='k')
    ax[0].grid(ls=':')
    ax[1].grid(ls=':')
    fig.tight_layout()
    fig.show()
