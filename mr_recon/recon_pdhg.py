"""example script on how to do MR recon with non-smooth prior using PDHG"""
import numpy as np

from phantoms import rod_phantom
from operators import MultiChannel3DCartesianMRAcquisitionModel, ComplexGradientOperator
from functionals import L2NormSquared, ComplexL1L2Norm
from algorithms import PDHG, sum_of_squares_reconstruction

import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------
# input parameters

n: int = 128
num_channels: int = 4
noise_level: float = 1.
seed: int = 0

num_iter: int = 50

data_norm = L2NormSquared()

prior_norm = ComplexL1L2Norm()
beta: float = 0.5
#prior_norm = L2NormSquared()
#beta: float = 5.

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

# do a sum of squares recon of the data
sos = sum_of_squares_reconstruction(noisy_data)

# setup the prior operator
prior_operator = ComplexGradientOperator(n, 3)

#----------------------------------------------------------------------------------------
# power iterations to estimate the norm of the complete operator (data + prior)
# for PDHG you should make sure that the norms of the individual operators are more or less the same

rimg = np.random.rand(*data_operator.x_shape)
for i in range(20):
    fwd1 = data_operator.forward(rimg)
    fwd2 = prior_operator.forward(rimg)

    rimg = data_operator.adjoint(fwd1) + prior_operator.adjoint(fwd2)
    pnsq = np.linalg.norm(rimg)
    rimg /= pnsq

op_norm = np.sqrt(pnsq)
print(op_norm)

#----------------------------------------------------------------------------------------
# run PDHG reconstruction

sigma = 0.99 / op_norm
tau = 0.99 / op_norm

reconstructor = PDHG(noisy_data, data_operator, data_norm, prior_operator,
                     prior_norm, beta, sigma, tau)

reconstructor.run(num_iter, calculate_cost=True)

#----------------------------------------------------------------------------------------
# show the results
ims = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.3 * np.percentile(x_true, 90))
ims_back = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.3 * np.percentile(sos, 90))

fig, ax = plt.subplots(3, 3, figsize=(8, 8))
ax[0, 0].imshow(np.linalg.norm(x_true, axis=-1)[..., n // 2], **ims)
ax[0, 1].imshow(x_true[..., n // 2, 0], **ims)
ax[0, 2].imshow(x_true[..., n // 2, 1], **ims)

ax[1, 0].imshow(sos[..., n // 2], **ims_back)
ax[1, 1].set_axis_off()
ax[1, 2].set_axis_off()

ax[2, 0].imshow(np.linalg.norm(reconstructor.x, axis=-1)[..., n // 2], **ims)
ax[2, 1].imshow(reconstructor.x[..., n // 2, 0], **ims)
ax[2, 2].imshow(reconstructor.x[..., n // 2, 1], **ims)

ax[0, 0].set_ylabel('ground truth')
ax[1, 0].set_ylabel('sum of squares')
ax[2, 0].set_ylabel('iterative w prior')

ax[0, 0].set_title('magnitude')
ax[0, 1].set_title('real part')
ax[0, 2].set_title('imag part')

fig.tight_layout()
fig.show()

# show the total, data and prior cost
iter = np.arange(1, reconstructor.cost.shape[0] + 1)
fig2, ax2 = plt.subplots(2, 3, figsize=(9, 6), sharex='row')
ax2[0, 0].plot(iter, reconstructor.cost)
ax2[0, 1].plot(iter, reconstructor.cost_data)
ax2[0, 2].plot(iter, reconstructor.cost_prior)
ax2[1, 0].plot(iter[-10:], reconstructor.cost[-10:])
ax2[1, 1].plot(iter[-10:], reconstructor.cost_data[-10:])
ax2[1, 2].plot(iter[-10:], reconstructor.cost_prior[-10:])
ax2[0, 0].set_title('total cost')
ax2[0, 1].set_title('data cost')
ax2[0, 2].set_title('prior cost')
for axx in ax2.ravel():
    axx.grid(ls=':')
for axx in ax2[-1, :]:
    axx.set_xlabel('iteration')
fig2.tight_layout()
fig2.show()

plt.show()