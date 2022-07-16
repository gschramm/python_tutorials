# minimal example to see if MLEM also converges with scalar sens image

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from typing import Optional, Union


class AcqModel:
    def __init__(self, A: np.ndarray) -> None:
        self.A = A

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self.A.T @ y


class MLEM:
    def __init__(self,
                 acq_model: AcqModel,
                 data: np.ndarray,
                 contamination: Optional[np.ndarray] = None,
                 scalar_sens_image: Optional[bool] = False) -> None:
        self.acq_model = acq_model
        self.data = data
        self.contamination = contamination

        # init recon
        self.init_image()

        # calc. sens image
        self.adjoint_ones = self.acq_model.adjoint(
            np.ones(self.acq_model.A.shape[0]))

        if scalar_sens_image:
            self.adjoint_ones = 0 * self.adjoint_ones + self.adjoint_ones.max()

        # init iteration counter
        self.iteration_counter = 0

    def init_image(self) -> None:
        # initialize recon with ones
        self.image = np.ones(self.acq_model.A.shape[1])
        self.iterates: list = []

    def update(self) -> None:
        exp = self.acq_model.forward(self.image)

        if self.contamination is not None:
            exp += self.contamination

        gradient = self.acq_model.adjoint((self.data - exp) / exp)
        step = self.image / self.adjoint_ones

        self.image += step * gradient

        self.iterates.append(self.image.copy())

    def run(self, niter: int) -> np.ndarray:
        for i in range(niter):
            self.update()
            self.iteration_counter += 1

        return np.array(self.iterates)


#----------------------------------------------------------------------------------------------
np.random.seed(1)

niter = 1000
A = np.random.rand(5, 3)
A[:, 1] *= 5
x_true = 10 * np.random.rand(A.shape[1])
#----------------------------------------------------------------------------------------------

model = AcqModel(A)

noise_free_data = model.forward(x_true)
contam = np.full(noise_free_data.shape, 5.3)

noise_free_data += contam

reconstructor_1 = MLEM(model,
                       noise_free_data,
                       contamination=contam,
                       scalar_sens_image=False)
reconstructor_2 = MLEM(model,
                       noise_free_data,
                       contamination=contam,
                       scalar_sens_image=True)

x1 = reconstructor_1.run(niter)
x2 = reconstructor_2.run(niter)

#----------------------------------------------------------------------------------------------
# show the solutions
fig, ax = plt.subplots(2,
                       x_true.shape[0],
                       figsize=(3 * x_true.shape[0], 6),
                       sharey='row')

for i in range(x_true.shape[0]):
    ax[0, i].plot(x1[:, i])
    ax[0, i].plot(x2[:, i])
    ax[0, i].axhline(x_true[i], color='k')
    ax[0, i].grid(ls=':')

    ax[1, i].loglog(np.abs(x1[:, i] - x_true[i]))
    ax[1, i].loglog(np.abs(x2[:, i] - x_true[i]))
    ax[1, i].grid(ls=':')

    ax[0, i].set_xlabel('iteration')
    ax[1, i].set_xlabel('iteration')

ax[0, 0].set_ylabel('recon. value')
ax[1, 0].set_ylabel('abs(recon value - g.t.)')

fig.tight_layout()
fig.show()
