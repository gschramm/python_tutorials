"""example script on how to do MR recon with non-smooth prior using PDHG"""
import numpy as np

from phantoms import rod_phantom
from operators import MultiChannel3DNonCartesianMRAcquisitionModel, ComplexGradientOperator
from functionals import L2NormSquared, ComplexL1L2Norm
from algorithms import PDHG

import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------
# input parameters

n: int = 64
num_channels: int = 1
noise_level: float = 0.
seed: int = 0

num_spokes: int = int(0.5 * (64**2))
num_samples_per_spoke: int = 128  # this must be an even number

num_iter: int = 300

data_norm = L2NormSquared()

prior_norm = ComplexL1L2Norm()
beta: float = 3e-4

#----------------------------------------------------------------------------------------
np.random.seed(seed)

# setup ground truth image and simulate data
# ground truth image
ph = rod_phantom(n=n)
x_true = np.stack([ph, np.zeros_like(ph)], axis=-1)

# setup "known" coil sensitivities - in real life they have to estimated from the data
#sens = generate_sensitivities(n, num_channels) / 2000
sens = np.ones((num_channels, n, n, n, 2)) / 20000
sens[..., 1] = 0

# setup the data operator for a 3D radial sequence
phis = 2 * np.pi * np.random.rand(num_spokes)
cos_thetas = 2 * np.random.rand(num_spokes) - 1
sin_thetas = np.sqrt(1 - cos_thetas**2)

kspace_sample_points = np.zeros((num_spokes * num_samples_per_spoke, 3))

r = np.linspace(-np.pi, np.pi, num_samples_per_spoke)
for isp in np.arange(num_spokes):
    offset = isp * num_samples_per_spoke
    kspace_sample_points[offset:(offset + num_samples_per_spoke),
                         0] = r * np.cos(phis[isp]) * sin_thetas[isp]
    kspace_sample_points[offset:(offset + num_samples_per_spoke),
                         1] = r * np.sin(phis[isp]) * sin_thetas[isp]
    kspace_sample_points[offset:(offset + num_samples_per_spoke),
                         2] = r * cos_thetas[isp]

data_operator = MultiChannel3DNonCartesianMRAcquisitionModel(
    n, num_channels, sens, kspace_sample_points)

# generate noise-free data
noise_free_data = data_operator.forward(x_true)

noisy_data = noise_free_data + noise_level * np.abs(
    noise_free_data).mean() * np.random.randn(*noise_free_data.shape)

# apply the adjoint of the data operator to the data
data_back = data_operator.adjoint(noisy_data)

# generate data that is corrected for sampling density of a sphere (times r^2)
noisy_data_sampling_corrected = noisy_data.copy()
for ch in range(num_channels):
    for i in range(2):
        noisy_data_sampling_corrected[ch, :, i] *= np.tile(r**2, num_spokes)

# apply the adjoint of the data operator to the data
data_back_sampling_corrected = data_operator.adjoint(
    noisy_data_sampling_corrected)

# setup the prior operator
prior_operator = ComplexGradientOperator(n, 3)

#----------------------------------------------------------------------------------------
# power iterations to estimate the norm of the complete operator (data + prior)
# for PDHG you should make sure that the norms of the individual operators are more or less the same

rimg = np.random.rand(*data_operator.x_shape)
for i in range(20):
    print(f'power iteration {i+1}')
    fwd1 = data_operator.forward(rimg)
    fwd2 = prior_operator.forward(rimg)

    rimg = data_operator.adjoint(fwd1) + prior_operator.adjoint(fwd2)
    pnsq = np.linalg.norm(rimg)
    rimg /= pnsq

op_norm = np.sqrt(pnsq)
print(op_norm)

#----------------------------------------------------------------------------------------
# run PDHG reconstruction

sigma = 0.01
tau = 0.99 / (sigma * (op_norm**2))

reconstructor = PDHG(noisy_data, data_operator, data_norm, prior_operator,
                     prior_norm, beta, sigma, tau)

reconstructor.run(num_iter, calculate_cost=True)

#----------------------------------------------------------------------------------------
# show the results
ims = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.3 * np.percentile(x_true, 90))
ims_back = dict(cmap=plt.cm.Greys_r,
                vmin=0,
                vmax=1.3 * np.percentile(data_back_sampling_corrected, 99))

fig, ax = plt.subplots(3, 3, figsize=(8, 8))
ax[0, 0].imshow(np.linalg.norm(x_true, axis=-1)[..., n // 2], **ims)
ax[0, 1].imshow(x_true[..., n // 2, 0], **ims)
ax[0, 2].imshow(x_true[..., n // 2, 1], **ims)

ax[1, 0].imshow(
    np.linalg.norm(data_back_sampling_corrected, axis=-1)[..., n // 2],
    **ims_back)
ax[1, 1].imshow(data_back_sampling_corrected[..., n // 2, 0], **ims_back)
ax[1, 2].imshow(data_back_sampling_corrected[..., n // 2, 1], **ims_back)

ax[2, 0].imshow(np.linalg.norm(reconstructor.x, axis=-1)[..., n // 2], **ims)
ax[2, 1].imshow(reconstructor.x[..., n // 2, 0], **ims)
ax[2, 2].imshow(reconstructor.x[..., n // 2, 1], **ims)

ax[0, 0].set_ylabel('ground truth')
ax[1, 0].set_ylabel('adjoint(corrected data)')
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