"""example script for MR reconstruction with smooth cost function using CG and LBFGS"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b, fmin_cg

from phantoms import rod_phantom, generate_sensitivities
from operators import MultiChannel3DCartesianMRAcquisitionModel, ComplexGradientOperator
from functionals import TotalCost, L2NormSquared
from algorithms import sum_of_squares_reconstruction

#----------------------------------------------------------------------------------------
# input parameters

n: int = 64
num_channels: int = 4
noise_level: float = 1.
seed: int = 0

num_iter: int = 50

data_norm = L2NormSquared()

prior_norm = L2NormSquared()
beta: float = 1.

#----------------------------------------------------------------------------------------
np.random.seed(seed)

# setup ground truth image and simulate data
# ground truth image
ph = rod_phantom(n=n)
x_true = np.stack([ph, np.zeros_like(ph)], axis=-1)

# setup "known" coil sensitivities - in real life they have to estimated from the data
sens = generate_sensitivities(n, num_channels)

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

ims = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.3 * np.percentile(x_true, 90))
ims_back = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.3 * np.percentile(sos, 90))

fig, ax = plt.subplots(4, 3, figsize=(6, 8))
ax[0, 0].imshow(np.linalg.norm(x_true, axis=-1)[..., n // 2], **ims)
ax[0, 1].imshow(x_true[..., n // 2, 0], **ims)
ax[0, 2].imshow(x_true[..., n // 2, 1], **ims)

ax[1, 0].imshow(sos[..., n // 2], **ims_back)
ax[1, 1].set_axis_off()
ax[1, 2].set_axis_off()

ax[2, 0].imshow(np.linalg.norm(recon_cg, axis=-1)[..., n // 2], **ims)
ax[2, 1].imshow(recon_cg[..., n // 2, 0], **ims)
ax[2, 2].imshow(recon_cg[..., n // 2, 1], **ims)

ax[3, 0].imshow(np.linalg.norm(recon_lbfgs, axis=-1)[..., n // 2], **ims)
ax[3, 1].imshow(recon_lbfgs[..., n // 2, 0], **ims)
ax[3, 2].imshow(recon_lbfgs[..., n // 2, 1], **ims)

ax[0, 0].set_ylabel('ground truth')
ax[1, 0].set_ylabel('sum of squares')
ax[2, 0].set_ylabel('iterative w prior CG')
ax[3, 0].set_ylabel('iterative w prior LBFGS')

ax[0, 0].set_title('magnitude')
ax[0, 1].set_title('real part')
ax[0, 2].set_title('imag part')

fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(np.arange(1, len(cost_cg) + 1)[3:], cost_cg[3:], label='CG')
ax2.plot(np.arange(1, len(cost_lbfgs) + 1)[3:], cost_lbfgs[3:], label='L-BFGS')
ax2.legend()
ax2.set_xlabel('iteration')
ax2.set_ylabel('cost')
ax2.grid(ls=':')
fig2.tight_layout()
fig2.show()

plt.show()