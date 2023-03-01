import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize


class MatrixOperator:

    def __init__(self, A: np.ndarray):
        self._A = A
        self._ishape = (A.shape[1], )
        self._oshape = (A.shape[0], )

    @property
    def ishape(self) -> tuple:
        return self._ishape

    @property
    def oshape(self) -> tuple:
        return self._oshape

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._A @ x

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self._A.T @ y


class NegativePoissonLogL:

    def __init__(self, model: MatrixOperator, data: np.ndarray,
                 contamination: np.ndarray):
        self._model = model
        self._data = data
        self._contamination = contamination

    def __call__(self, x: np.ndarray) -> float:
        expectation = self._model.forward(x) + self._contamination
        return np.sum(expectation - self._data * np.log(expectation))


class MLEMPoisson:

    def __init__(self, model: MatrixOperator, data: np.ndarray,
                 contamination: np.ndarray):
        self._model = model
        self._data = data
        self._contamination = contamination

        self._sens = self._model.adjoint(np.ones(self._model.oshape))

        self._x = np.ones(self._model.ishape)

        self._cost_function = NegativePoissonLogL(self._model, self._data,
                                                  self._contamination)

        self._cost = []

    @property
    def x(self) -> np.ndarray:
        return self._x

    @x.setter
    def x(self, value: np.ndarray):
        self._x = value

    @property
    def model(self) -> MatrixOperator:
        return self._model

    @property
    def cost(self) -> list:
        return self._cost

    def multiplicative_update(self):
        expectation = self._model.forward(self._x) + self._contamination
        ratio = self._data / expectation
        ratio_back = self._model.adjoint(ratio)

        update = (ratio_back / self._sens)

        return update

    def run(self, num_iterations: int):
        for _ in range(num_iterations):
            self._x *= self.multiplicative_update()
            self._cost.append(self._cost_function(self._x))

    def brute_force(self) -> tuple[np.ndarray, float]:
        res = minimize(self._cost_function, self._x)
        return res.x, res.fun


class MulticlassMLEM:

    def __init__(self, MLEM1: MLEMPoisson, MLEM2: MLEMPoisson):
        self._MLEM1 = MLEM1
        self._MLEM2 = MLEM2

        self._x = np.ones(self._MLEM1.model.ishape)

    @property
    def x(self) -> np.ndarray:
        return self._x

    def run(self, num_iterations: int):
        for _ in range(num_iterations):
            u1 = self._MLEM1.multiplicative_update()
            u2 = self._MLEM1.multiplicative_update()

            self._x *= (u1 + u2) / 2


if __name__ == '__main__':
    np.random.seed(3)
    num_iter = 500
    split = 3
    noise = False
    A = np.random.rand(9, 2)
    model = MatrixOperator(A)

    model1 = MatrixOperator(A[:split, :])
    model2 = MatrixOperator(A[split:, :])

    #---------------------------------------------------------------

    x_true = 30 * np.random.rand(*model.ishape)
    noise_free_data = model.forward(x_true)
    contamination = np.full(model.oshape, 0.05 * noise_free_data.mean())

    if noise:
        data = np.random.poisson(noise_free_data + contamination)
    else:
        data = np.random.poisson(noise_free_data + contamination)

    data1 = data[:split]
    data2 = data[split:]

    contamination1 = contamination[:split]
    contamination2 = contamination[split:]

    # MLEM using complete model
    alg = MLEMPoisson(model, data, contamination)
    alg.run(num_iter)
    x_ref, cost_ref = alg.brute_force()

    # MLEM using first part of data
    alg1 = MLEMPoisson(model1, data1, contamination1)
    alg1.run(num_iter)

    # MLEM using second part of data
    alg2 = MLEMPoisson(model2, data2, contamination2)
    alg2.run(num_iter)

    # "MultiClass" MLEM
    alg3 = MulticlassMLEM(alg1, alg2)
    alg3.run(num_iter * 10)

    ##------------------------------------------------------------------
    ## plots
    #it = np.arange(1, num_iter + 1)
    #fig, ax = plt.subplots()
    #ax.loglog(it, np.abs(alg.cost))
    #ax.axhline(np.abs(cost_ref), color='r', linestyle='--')
    #ax.grid(ls=':')
    #fig.tight_layout()
    #fig.show()