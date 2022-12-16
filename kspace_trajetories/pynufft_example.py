"""demo to show how to simulate non-uniform kspace data and on how to reconstruct them via conjugate gradient"""

import numpy as np
from scipy.optimize import fmin_cg, fmin_l_bfgs_b
import matplotlib.pyplot as plt

from operators import MultiChannelStackedNonCartesianMRAcquisitionModel, GradientOperator
from kspace_trajectories import radial_2d_golden_angle
from functions import SquaredL2Norm

if __name__ == '__main__':
    recon_shape = (4, 256, 256)
    num_iterations = 20
    undersampling_factor = 32
    beta = 1e6

    num_spokes = int(recon_shape[1] * np.pi / 2) // undersampling_factor
    num_samples_per_spoke = recon_shape[0]
    print(f'number of spokes {num_spokes}')

    print('loading image')
    x = np.load('../mr_recon/xcat_vol.npz')['arr_0'].reshape(
        1024, 1024, 1024).astype(np.float32)[::2, ::2, ::2]

    # normalize x to have max 1
    x /= x.max()

    # swap axes to have sagittal axis in front
    x = np.swapaxes(x, 0, 2)

    # select one sagital slice
    if len(recon_shape) == 2:
        start0 = x.shape[0] // 2
        end0 = start0 + 1
        start1 = x.shape[0] // 2 - recon_shape[0] // 2
        end1 = start1 + recon_shape[0]
        start2 = x.shape[1] // 2 - recon_shape[1] // 2
        end2 = start2 + recon_shape[1]
    else:
        start0 = x.shape[0] // 2 - recon_shape[0] // 2
        end0 = start0 + recon_shape[0]
        start1 = x.shape[1] // 2 - recon_shape[1] // 2
        end1 = start1 + recon_shape[1]
        start2 = x.shape[2] // 2 - recon_shape[2] // 2
        end2 = start2 + recon_shape[2]

    x = x[start0:end0, start1:end1, start2:end2]

    x = x.astype(np.complex64)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
print('setting up operators')

# setup the kspace sample points for the operator acting on the high res image
# here we only take the center which is why kmax is only a fraction np.pi
# reconstructing on a lower res grid needs only the "center part" of kspace

kspace_points = radial_2d_golden_angle(num_spokes, recon_shape[-1])

sens = np.expand_dims(np.ones(recon_shape).astype(x.dtype), 0)

data_operator = MultiChannelStackedNonCartesianMRAcquisitionModel(
    recon_shape, sens, kspace_points, device_number=0)

data = data_operator.forward(x)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# setup the cost functions
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

data_distance = SquaredL2Norm(xp=np, shift=data)
prior_operator = GradientOperator(recon_shape, xp=np, dtype=x.dtype)
prior_norm = SquaredL2Norm(xp=np, scale=beta)

data_fidelity = lambda z: data_distance(
    data_operator.forward(data_operator.unravel_pseudo_complex(z)))
data_fidelity_gradient = lambda z: data_operator.ravel_pseudo_complex(
    data_operator.adjoint(
        data_distance.gradient(
            data_operator.forward(data_operator.unravel_pseudo_complex(z)))))

prior = lambda z: prior_norm(
    prior_operator.forward(prior_operator.unravel_pseudo_complex(z)))
prior_gradient = lambda z: prior_operator.ravel_pseudo_complex(
    prior_operator.adjoint(
        prior_norm.gradient(
            prior_operator.forward(prior_operator.unravel_pseudo_complex(z)))))

loss = lambda z: data_fidelity(z) + prior(z)
loss_gradient = lambda z: data_fidelity_gradient(z) + prior_gradient(z)

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# setup a loss function and run a cg reconstruction
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
print('running the recon')

x0 = np.zeros(2 * x.size, dtype=x.real.dtype)

res = fmin_cg(loss,
              x0,
              fprime=loss_gradient,
              maxiter=num_iterations,
              retall=True)
recon = data_operator.unravel_pseudo_complex(res[0])

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# visualizations
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

ims = dict(cmap=plt.cm.Greys_r, origin='lower')
fig, ax = plt.subplots(2, 3, figsize=(3 * 4, 2 * 4))
ax[0, 0].imshow(np.abs(x[0, ...]).T, **ims)
ax[1, 0].imshow(np.abs(recon[0, ...]).T, **ims)
ax[0, 1].imshow(np.abs(x[x.shape[0] // 2, ...]).T, **ims)
ax[1, 1].imshow(np.abs(recon[recon_shape[0] // 2, ...]).T, **ims)
ax[0, 2].imshow(np.abs(x[-1, ...]).T, **ims)
ax[1, 2].imshow(np.abs(recon[-1, ...]).T, **ims)
fig.tight_layout()
fig.show()