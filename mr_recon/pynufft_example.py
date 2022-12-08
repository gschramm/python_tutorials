"""demo to show how to simulate non-uniform kspace data and on how to reconstruct them via conjugate gradient"""

import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt

from operators import MultiChannelNonCartesianMRAcquisitionModel, ComplexGradientOperator
from functionals import TotalCost, L2NormSquared
from kspace_trajectories import radial_2d_golden_angle

if __name__ == '__main__':
    recon_shape = (256, 256)
    num_iterations = 100
    undersampling_factor = 8

    num_spokes = int(recon_shape[1] * np.pi / 2) // undersampling_factor
    num_samples_per_spoke = recon_shape[1]

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
    # select one sagital slice
    x = x[:, :, 512]

    kspace_scaling_factors = np.array(x.shape) / np.array(recon_shape)
    scaling_factor = (np.array(x.shape) / np.array(recon_shape)).prod()

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

    kspace_sample_points_high_res = radial_2d_golden_angle(
        num_spokes,
        num_samples_per_spoke,
        kmax=np.pi / kspace_scaling_factors[0]).astype(np.float32)

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

    # reshape the flattened cg recon
    recon_cg = recon_cg.reshape(recon_shape + (2, ))

    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------
    # visualizations
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------

    ims = dict(cmap=plt.cm.Greys_r)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.linalg.norm(x, axis=-1)[::-1, :], **ims)
    ax[1].imshow(np.linalg.norm(recon_cg, axis=-1)[::-1, :], **ims)
    fig.tight_layout()
    fig.show()
