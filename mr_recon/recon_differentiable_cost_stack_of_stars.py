"""example script on how to do MR recon with non-smooth prior using PDHG"""
import numpy as np
from scipy.optimize import fmin_cg

from phantoms import rod_phantom
from operators import MultiChannel3DStackOfStarsMRAcquisitionModel, ComplexGradientOperator
from functionals import L2NormSquared, TotalCost

import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------
# input parameters

n: int = 128
n2: int = 15
num_channels: int = 1
noise_level: float = 0.1
seed: int = 0

num_spokes: int = 32
num_samples_per_spoke: int = 2 * n - 1

num_iter: int = 200

data_norm = L2NormSquared()

prior_norm = L2NormSquared()
beta: float = 1e2

#----------------------------------------------------------------------------------------
np.random.seed(seed)

# setup ground truth image and simulate data
# ground truth image
ph = np.swapaxes(rod_phantom(n=n)[:, :, (n // 2):(n // 2 + n2)], 0, 2)
x_true = np.stack([ph, np.zeros_like(ph)], axis=-1)

# setup "known" coil sensitivities - in real life they have to estimated from the data
sens = np.ones((num_channels, n2, n, n, 2)) / 100
sens[..., 1] = 0

# choose random stack numbers

data_operator = MultiChannel3DStackOfStarsMRAcquisitionModel(
    x_shape=(n2, n, n),
    coil_sensitivities=sens,
    spoke_angles=np.pi * np.random.rand(num_spokes),
    num_samples_per_spoke=num_samples_per_spoke)

# generate noise-free data
noise_free_data = data_operator.forward(x_true)

noisy_data = noise_free_data + noise_level * np.abs(
    noise_free_data).mean() * np.random.randn(*noise_free_data.shape)

# apply the adjoint of the data operator to the data
data_back = data_operator.adjoint(noisy_data)

# generate data that is corrected for sampling density of a star (times |rho|)
noisy_data_sampling_corrected = noisy_data.copy()
rho = np.sqrt(data_operator.kspace_sample_points[:, 1]**2 +
              data_operator.kspace_sample_points[:, 2]**2)

for ch in range(num_channels):
    for i in range(2):
        noisy_data_sampling_corrected[ch, :, i] *= rho

# apply the adjoint of the data operator to the data
data_back_sampling_corrected = data_operator.adjoint(
    noisy_data_sampling_corrected)

# setup the prior operator
prior_operator = ComplexGradientOperator((n2, n, n), 3)

#----------------------------------------------------------------------------------------
# run CG recon
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

#----------------------------------------------------------------------------------------
# show the results
ims = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.3 * np.percentile(x_true, 90))
ims_back = dict(cmap=plt.cm.Greys_r,
                vmin=0,
                vmax=1.3 * np.percentile(data_back_sampling_corrected, 99))

fig, ax = plt.subplots(3, 3, figsize=(8, 8))
ax[0, 0].imshow(np.linalg.norm(x_true, axis=-1)[n2 // 2, ...], **ims)
ax[0, 1].imshow(x_true[n2 // 2, ..., 0], **ims)
ax[0, 2].imshow(x_true[n2 // 2, ..., 1], **ims)

ax[1, 0].imshow(
    np.linalg.norm(data_back_sampling_corrected, axis=-1)[n2 // 2, ...],
    **ims_back)
ax[1, 1].imshow(data_back_sampling_corrected[n2 // 2, ..., 0], **ims_back)
ax[1, 2].imshow(data_back_sampling_corrected[n2 // 2, ..., 1], **ims_back)

ax[2, 0].imshow(np.linalg.norm(recon_cg, axis=-1)[n2 // 2, ...], **ims)
ax[2, 1].imshow(recon_cg[n2 // 2, ..., 0], **ims)
ax[2, 2].imshow(recon_cg[n2 // 2, ..., 1], **ims)

ax[0, 0].set_ylabel('ground truth')
ax[1, 0].set_ylabel('adjoint(density corr. data)')
ax[2, 0].set_ylabel('iterative w prior')

ax[0, 0].set_title('magnitude')
ax[0, 1].set_title('real part')
ax[0, 2].set_title('imag part')

fig.tight_layout()
fig.show()

# show the total, data and prior cost
iter = np.arange(1, len(cost_cg) + 1)
fig2, ax2 = plt.subplots(1, 2, figsize=(8, 4))
ax2[0].plot(iter, cost_cg)
ax2[1].plot(iter[-10:], cost_cg[-10:])
for axx in ax2:
    axx.grid(ls=':')
    axx.set_xlabel('iteration')
fig2.tight_layout()
fig2.show()

fig3 = data_operator.show_kspace_sample_points()