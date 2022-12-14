"""demo to show how to simulate non-uniform kspace data and on how to reconstruct them via conjugate gradient"""

import numpy as np
from scipy.optimize import fmin_cg, fmin_l_bfgs_b
import matplotlib.pyplot as plt

from operators import MultiChannelNonCartesianMRAcquisitionModel, ComplexGradientOperator
from functionals import TotalCost, L2NormSquared
from kspace_trajectories import radial_2d_golden_angle, stack_of_2d_golden_angle

if __name__ == '__main__':
    recon_shape = (256, 256)
    num_iterations = 1000
    undersampling_factor = 8

    num_spokes = int(recon_shape[1] * np.pi / 2) // undersampling_factor
    num_samples_per_spoke = recon_shape[0]

    print(f'number of spokes {num_spokes}')

    spoke_angles = (np.arange(num_spokes) * (2 * np.pi /
                                             (1 + np.sqrt(5)))) % (2 * np.pi)

    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    # generate pseudo-complex image and add fake 3rd dimension
    # the image is generated on a "high-res" (1024,1024,1024) which is important
    # to approximate the continous FT using DFT
    # the reconstruction will be performed on a lower resolution grid
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    print('loading image')

    x = np.load('xcat_vol.npz')['arr_0'].reshape(1024, 1024,
                                                 1024).astype(np.float32)

    # normalize x to have max 1
    x /= x.max()

    # swap axes to have sagittal axis in front
    x = np.swapaxes(x, 0, 2)

    # select one sagital slice
    if len(recon_shape) == 2:
        x = x[512, :, :]
    elif len(recon_shape) == 3:
        # take only every second element in 3D to save memory
        x = x[::4, ::4, ::4]
    else:
        raise ValueError

    kspace_scaling_factors = np.array(x.shape) / np.array(recon_shape)
    scaling_factor = kspace_scaling_factors.prod()

    # convert to pseudo complex 3D array by adding a 0 imaginary part
    x = np.stack([x, np.zeros_like(x)], axis=-1)

    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    # setup a forward model that acts on the high resolution ground truth image
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    print('setting up operators')

    # setup the kspace sample points for the operator acting on the high res image
    # here we only take the center which is why kmax is only a fraction np.pi
    # reconstructing on a lower res grid needs only the "center part" of kspace

    if len(recon_shape) == 2:
        kspace_sample_points_high_res = radial_2d_golden_angle(
            num_spokes,
            num_samples_per_spoke,
            kmax=np.pi / kspace_scaling_factors[-1]).astype(np.float32)
    elif len(recon_shape) == 3:
        kspace_sample_points_high_res = stack_of_2d_golden_angle(
            num_stacks=recon_shape[0],
            kzmax=np.pi / kspace_scaling_factors[0],
            num_spokes=num_spokes,
            num_samples_per_spoke=num_samples_per_spoke,
            kmax=np.pi / kspace_scaling_factors[-1]).astype(np.float32)

    data_simulation_operator = MultiChannelNonCartesianMRAcquisitionModel(
        x.shape[:-1], np.expand_dims(np.ones(x.shape, dtype=x.dtype), 0),
        kspace_sample_points_high_res)

    data = data_simulation_operator.forward(x)

    # setup a forward model that acts on a lower resolution image
    kspace_sample_points_low_res = kspace_sample_points_high_res.copy()

    for i in range(kspace_scaling_factors.shape[0]):
        kspace_sample_points_low_res[:, i] *= kspace_scaling_factors[i]

    data_recon_operator = MultiChannelNonCartesianMRAcquisitionModel(
        recon_shape,
        np.expand_dims(np.ones(recon_shape + (2, ), dtype=x.dtype), 0),
        kspace_sample_points_low_res,
        scaling_factor=scaling_factor)

    prior_operator = ComplexGradientOperator(recon_shape, len(recon_shape))

    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    # setup a loss function and run a cg reconstruction
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    print('running the recon')

    # switch to flat input and output need that we need for fmin_cg
    data_recon_operator.flat_mode = True
    prior_operator.flat_mode = True

    data_norm = L2NormSquared()
    prior_norm = L2NormSquared()
    beta = 0.

    x0 = np.zeros(np.prod(recon_shape + (2, )), dtype=x.dtype)

    loss = TotalCost(data.ravel(),
                     data_recon_operator,
                     data_norm,
                     prior_operator,
                     prior_norm,
                     beta=beta)

    # cg recon method
    recon_cg = fmin_cg(loss, x0, fprime=loss.gradient, maxiter=num_iterations)
    recon_cg = recon_cg.reshape(recon_shape + (2, ))

    # LBFGS recon method
    recon_lb = fmin_l_bfgs_b(loss,
                             x0,
                             fprime=loss.gradient,
                             maxiter=num_iterations,
                             disp=2)
    recon_lb = recon_lb[0].reshape(recon_shape + (2, ))

    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    # visualizations
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------

    ims = dict(cmap=plt.cm.Greys_r, origin='lower')
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    if len(recon_shape) == 2:
        ax[0].imshow(np.linalg.norm(x, axis=-1).T, **ims)
        ax[1].imshow(np.linalg.norm(recon_cg, axis=-1).T, **ims)
        ax[2].imshow(np.linalg.norm(recon_lb, axis=-1).T, **ims)
    if len(recon_shape) == 3:
        ax[0].imshow(np.linalg.norm(x, axis=-1)[x.shape[0] // 2, ...].T, **ims)
        ax[1].imshow(
            np.linalg.norm(recon_cg, axis=-1)[recon_shape[0] // 2, ...].T,
            **ims)
        ax[1].imshow(
            np.linalg.norm(recon_lb, axis=-1)[recon_shape[0] // 2, ...].T,
            **ims)
    fig.tight_layout()
    fig.show()
