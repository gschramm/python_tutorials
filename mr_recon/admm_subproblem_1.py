"""script that shows how to solve ADMM subproblem 1:
   
   argmin_x L2NormSquared(MR_operator x - data) + beta * L2NormSquared(x - prior_image)
   
   using different optimizers

   Note: L2NormSquared = 0.5 * \sum_i x_i^2  (factor 0.5 included)
"""
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b, fmin_cg

from phantoms import rod_phantom, generate_sensitivities
from operators import IdentityOperator, MultiChannel3DCartesianMRAcquisitionModel, ComplexGradientOperator, IdentityOperator
from functionals import TotalCost, L2NormSquared
from algorithms import sum_of_squares_reconstruction, PDHG

#----------------------------------------------------------------------------------------
# input parameters

n: int = 64
num_channels: int = 4
noise_level: float = 1.
seed: int = 0

num_iter: int = 50

data_norm = L2NormSquared()

prior_norm = L2NormSquared()
prior_operator = IdentityOperator((n, n, n, 2))
beta: float = 2.

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

#----------------------------------------------------------------------------------------
# run the recon with prior

# define the prior image (in the prior we penalize the norm(prior_ operator x - prior_image))
# here we just use a arbitrery image with ones in the real part and zeros in the imaginary part
prior_image = np.zeros(x_true.shape)
prior_image[..., 0] = 1

# the total cost function that has __call__(x) and gradient(x)
cost = TotalCost(noisy_data, data_operator, data_norm, prior_operator,
                 prior_norm, beta, prior_image)

# initial recon
x0 = np.zeros(data_operator.x_shape).ravel()

# optimization using conjugate gradient optimizer
cost_cg = []
t0 = time.time()
res_cg = fmin_cg(cost,
                 x0=x0,
                 fprime=cost.gradient,
                 maxiter=num_iter,
                 full_output=True,
                 callback=lambda x: cost_cg.append(cost(x)))
t1 = time.time()

recon_cg = res_cg[0].reshape(data_operator.x_shape)
final_cost_cg = res_cg[1]

# optimization using L_BFGS
cost_lbfgs = []
t2 = time.time()
res_lbfgs = fmin_l_bfgs_b(cost,
                          x0=x0,
                          fprime=cost.gradient,
                          maxiter=num_iter,
                          disp=1,
                          callback=lambda x: cost_lbfgs.append(cost(x)))
t3 = time.time()

recon_lbfgs = res_lbfgs[0].reshape(data_operator.x_shape)
final_cost_lbfgs = res_lbfgs[1]

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
# optimization using PDHG

sigma = 0.99 / op_norm
tau = 0.99 / op_norm

pdhg = PDHG(noisy_data,
            data_operator,
            data_norm,
            prior_operator,
            prior_norm,
            beta,
            sigma,
            tau,
            prior_image=prior_image)

t4 = time.time()
pdhg.run(num_iter, calculate_cost=True)
t5 = time.time()

#----------------------------------------------------------------------------------------
# show the results

print('')
print(f'time CG {t1-t0}')
print(f'time LBFGS-G {t3-t2}')
print(f'time PDHG {t5-t4}')

ims = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.3 * np.percentile(x_true, 90))
ims_back = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.3 * np.percentile(sos, 90))

fig, ax = plt.subplots(5, 3, figsize=(6, 10))
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

ax[4, 0].imshow(np.linalg.norm(pdhg.x, axis=-1)[..., n // 2], **ims)
ax[4, 1].imshow(pdhg.x[..., n // 2, 0], **ims)
ax[4, 2].imshow(pdhg.x[..., n // 2, 1], **ims)

ax[0, 0].set_ylabel('ground truth')
ax[1, 0].set_ylabel('sum of squares')
ax[2, 0].set_ylabel('iterative w prior CG')
ax[3, 0].set_ylabel('iterative w prior LBFGS')
ax[4, 0].set_ylabel('iterative w prior PDHG')

ax[0, 0].set_title('magnitude')
ax[0, 1].set_title('real part')
ax[0, 2].set_title('imag part')

fig.tight_layout()
fig.show()

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(np.arange(1, len(cost_cg) + 1), cost_cg, label='CG')
ax2.plot(np.arange(1, len(cost_lbfgs) + 1), cost_lbfgs, label='L-BFGS')
ax2.plot(np.arange(1, num_iter + 1), pdhg.cost, label='PDHG')
ax2.legend()
ax2.set_xlabel('iteration')
ax2.set_ylabel('cost')
ax2.grid(ls=':')
ax2.set_ylim(
    min([cost_cg[-1], cost_lbfgs[-1], pdhg.cost[-1]]) - 0.05 *
    (max(cost_cg[3:]) - cost_cg[-1]), max(cost_cg[3:]))
fig2.tight_layout()
fig2.show()

plt.show()