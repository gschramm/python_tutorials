"""demo to show how to simulate non-uniform kspace data and on how to reconstruct them via conjugate gradient"""

import numpy as np
from scipy.optimize import fmin_cg, fmin_l_bfgs_b
import matplotlib.pyplot as plt

from operators import MultiChannelStackedNonCartesianMRAcquisitionModel, GradientOperator
from kspace_trajectories import radial_2d_golden_angle

if __name__ == '__main__':
    recon_shape = (32, 256, 256)
    num_iterations = 20
    undersampling_factor = 8

    num_spokes = int(recon_shape[1] * np.pi / 2) // undersampling_factor
    num_samples_per_spoke = recon_shape[0]
    print(f'number of spokes {num_spokes}')

    print('loading image')
    x = np.load('../mr_recon/xcat_vol.npz')['arr_0'].reshape(
        1024, 1024, 1024).astype(np.float32)

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

sens = np.expand_dims(np.zeros(recon_shape).astype(x.dtype), 0)

data_operator = MultiChannelStackedNonCartesianMRAcquisitionModel(
    recon_shape, sens, kspace_points, device_number=0)

prior_operator = GradientOperator(recon_shape, xp=np, dtype=x.dtype)

data = data_operator.forward(x)
#    #-----------------------------------------------------------------------------
#    #-----------------------------------------------------------------------------
#    # setup a loss function and run a cg reconstruction
#    #-----------------------------------------------------------------------------
#    #-----------------------------------------------------------------------------
#    print('running the recon')
#
#    # switch to flat input and output need that we need for fmin_cg
#    data_recon_operator.flat_mode = True
#    prior_operator.flat_mode = True
#
#    data_norm = L2NormSquared()
#    prior_norm = L2NormSquared()
#    beta = 0.
#
#    x0 = np.zeros(np.prod(recon_shape + (2, )), dtype=x.dtype)
#
#    loss = TotalCost(data.ravel(),
#                     data_recon_operator,
#                     data_norm,
#                     prior_operator,
#                     prior_norm,
#                     beta=beta)
#
#    # cg recon method
#    recon_cg = fmin_cg(loss, x0, fprime=loss.gradient, maxiter=num_iterations)
#    recon_cg = recon_cg.reshape(recon_shape + (2, ))
#
#    # LBFGS recon method
#    recon_lb = fmin_l_bfgs_b(loss,
#                             x0,
#                             fprime=loss.gradient,
#                             maxiter=num_iterations,
#                             disp=2)
#    recon_lb = recon_lb[0].reshape(recon_shape + (2, ))
#
#    #-----------------------------------------------------------------------------
#    #-----------------------------------------------------------------------------
#    # visualizations
#    #-----------------------------------------------------------------------------
#    #-----------------------------------------------------------------------------
#
#    ims = dict(cmap=plt.cm.Greys_r, origin='lower')
#    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
#    if len(recon_shape) == 2:
#        ax[0].imshow(np.linalg.norm(x, axis=-1).T, **ims)
#        ax[1].imshow(np.linalg.norm(recon_cg, axis=-1).T, **ims)
#        ax[2].imshow(np.linalg.norm(recon_lb, axis=-1).T, **ims)
#    if len(recon_shape) == 3:
#        ax[0].imshow(np.linalg.norm(x, axis=-1)[x.shape[0] // 2, ...].T, **ims)
#        ax[1].imshow(
#            np.linalg.norm(recon_cg, axis=-1)[recon_shape[0] // 2, ...].T,
#            **ims)
#        ax[1].imshow(
#            np.linalg.norm(recon_lb, axis=-1)[recon_shape[0] // 2, ...].T,
#            **ims)
#    fig.tight_layout()
#    fig.show()
#