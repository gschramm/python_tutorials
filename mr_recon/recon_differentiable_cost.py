# TODO: - SOS recon

import numpy as np

from phantoms import rod_phantom
from operators import MultiChannel3DCartesianMRAcquisitionModel, ComplexGradientOperator
from norms import L2NormSquared

import matplotlib.pyplot as plt

from scipy.optimize import fmin_l_bfgs_b, fmin_cg

#----------------------------------------------------------------------------------------
from operators import LinearOperator
from norms import SmoothNorm
import numpy.typing as npt


class TotalCost:
    """ total cost consisting of data fidelity and prior"""

    def __init__(self, data: npt.NDArray, data_operator: LinearOperator,
                 data_norm: SmoothNorm, prior_operator: LinearOperator,
                 prior_norm: SmoothNorm, beta: float) -> None:

        self._data = data

        self._data_operator = data_operator
        self._data_norm = data_norm

        self._prior_operator = prior_operator
        self._prior_norm = prior_norm

        self._beta = beta

    def __call__(self, x: npt.NDArray) -> float:
        input_shape = x.shape
        # reshaping is necessary since the scipy optimizers only handle 1D arrays
        x = x.reshape(self._data_operator.x_shape)

        cost = self._data_norm(self._data_operator.forward(x) -
                               self._data) + self._beta * self._prior_norm(
                                   self._prior_operator.forward(x))

        x = x.reshape(input_shape)

        return cost

    def gradient(self, x: npt.NDArray) -> npt.NDArray:
        input_shape = x.shape
        # reshaping is necessary since the scipy optimizers only handle 1D arrays
        x = x.reshape(self._data_operator.x_shape)

        data_grad = self._data_operator.adjoint(
            self._data_norm.gradient(
                self._data_operator.forward(x) - self._data))
        prior_grad = self._beta * self._prior_operator.adjoint(
            self._prior_norm.gradient(self._prior_operator.forward(x)))

        x = x.reshape(input_shape)

        return (data_grad + prior_grad).reshape(input_shape)


#----------------------------------------------------------------------------------------
# input parameters

n: int = 64
num_channels: int = 4
noise_level: float = 1.
seed: int = 0

num_iter: int = 50

data_norm = L2NormSquared()

prior_norm = L2NormSquared()
beta: float = 5.

#----------------------------------------------------------------------------------------
np.random.seed(seed)

# setup ground truth image and simulate data
# ground truth image
ph = rod_phantom(n=n)
x_true = np.stack([ph, np.zeros_like(ph)], axis=-1)

# setup "perfect" coil sensitivities - this is not realistic and should be improved
sens = np.ones((num_channels, n, n, n, 2))
sens[..., 1] = 0

# setup the data model
data_operator = MultiChannel3DCartesianMRAcquisitionModel(
    n, num_channels, sens)

# generate noise-free data
noise_free_data = data_operator.forward(x_true)

noisy_data = noise_free_data + noise_level * x_true.mean() * np.random.randn(
    *data_operator.x_shape)

# apply the adjoint of the data operator to the data
data_back = data_operator.adjoint(noisy_data)

# setup the prior operator
prior_operator = ComplexGradientOperator(n, 3)

#----------------------------------------------------------------------------------------
# run the recon

# the total cost function that has __call__(x) and gradient(x)
cost = TotalCost(noisy_data, data_operator, data_norm, prior_operator,
                 prior_norm, beta)

# initial recon
x0 = np.zeros(data_operator.x_shape).ravel()

# recon using conjugate gradient optimizer
cost_cg = []
res_cg = fmin_cg(cost,
                 x0=x0,
                 fprime=cost.gradient,
                 maxiter=num_iter,
                 full_output=True,
                 callback=lambda x: cost_cg.append(cost(x)))

recon_cg = res_cg[0].reshape(data_operator.x_shape)
final_cost_cg = res_cg[1]

# recon using L_BFGS
cost_lbfgs = []
res_lbfgs = fmin_l_bfgs_b(cost,
                          x0=x0,
                          fprime=cost.gradient,
                          maxiter=num_iter,
                          disp=1,
                          callback=lambda x: cost_lbfgs.append(cost(x)))

recon_lbfgs = res_lbfgs[0].reshape(data_operator.x_shape)
final_cost_lbfgs = res_lbfgs[1]

#----------------------------------------------------------------------------------------
# show the results

ims = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.2 * x_true.max())
ims_back = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.2 * data_back.max())

fig, ax = plt.subplots(4, 3, figsize=(6, 8))
ax[0, 0].imshow(x_true[..., n // 2, 0], **ims)
ax[0, 1].imshow(x_true[..., n // 2, 1], **ims)
ax[0, 2].imshow(np.linalg.norm(x_true, axis=-1)[..., n // 2], **ims)
ax[1, 0].imshow(data_back[..., n // 2, 0], **ims_back)
ax[1, 1].imshow(data_back[..., n // 2, 1], **ims_back)
ax[1, 2].imshow(np.linalg.norm(data_back, axis=-1)[..., n // 2], **ims_back)
ax[2, 0].imshow(recon_cg[..., n // 2, 0], **ims)
ax[2, 1].imshow(recon_cg[..., n // 2, 1], **ims)
ax[2, 2].imshow(np.linalg.norm(recon_cg, axis=-1)[..., n // 2], **ims)
ax[3, 0].imshow(recon_lbfgs[..., n // 2, 0], **ims)
ax[3, 1].imshow(recon_lbfgs[..., n // 2, 1], **ims)
ax[3, 2].imshow(np.linalg.norm(recon_lbfgs, axis=-1)[..., n // 2], **ims)

ax[0, 0].set_ylabel('ground truth')
ax[1, 0].set_ylabel('adjoint(data)')
ax[2, 0].set_ylabel('iterative w prior CG')
ax[3, 0].set_ylabel('iterative w prior LBFGS')

ax[0, 0].set_title('real part')
ax[0, 1].set_title('imag part')
ax[0, 2].set_title('magnitude')

fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots()
ax2.plot(cost_cg, label='CG')
ax2.plot(cost_lbfgs, label='L-BFGS')
ax2.legend()
ax2.set_xlabel('iteration')
ax2.set_ylabel('cost')
fig2.tight_layout()
fig2.show()

plt.show()