"""script that shows how to setup NUFFT operators with different image grid sizes"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
from scipy.ndimage import gaussian_filter

import pynufft

from operators import real_view_of_complex_array, complex_view_of_real_array


class NUFFTOperator:

    def __init__(self,
                 kspace_coordinates,
                 image_shape,
                 oversampled_fft_shape,
                 interpolation_shape=None,
                 scale_factor=1.):

        self._kspace_coordinates = kspace_coordinates
        self._ndim = self._kspace_coordinates.shape[1]
        self._image_shape = image_shape
        self._oversampled_fft_shape = oversampled_fft_shape

        if interpolation_shape is None:
            interpolation_shape = self._ndim * (6, )
        else:
            self._interpolation_shape = interpolation_shape

        self._nufft = pynufft.NUFFT()
        self._nufft.plan(self._kspace_coordinates, self._image_shape,
                         self._oversampled_fft_shape,
                         self._interpolation_shape)

        self._adjoint_scaling_factor = np.prod(self._oversampled_fft_shape)

        self._kmask = (np.linalg.norm(self._kspace_coordinates, axis=1) <=
                       np.pi)

        self._scale_factor = scale_factor

    @property
    def kmask(self):
        return self._kmask

    def forward(self, x):
        return self._nufft.forward(x) * self._kmask * self._scale_factor

    def adjoint(self, y):
        return self._nufft.adjoint(
            y * self._kmask *
            self._scale_factor) * self._adjoint_scaling_factor


class DataFidelityLoss:

    def __init__(self, data, operator, real_input, flat_input, image_shape):
        self._data = data
        self._operator = operator
        self._real_input = real_input
        self._flat_input = flat_input
        self._image_shape = image_shape

    def _diff(self, x):
        exp = self._operator.forward(x)
        return (exp - self._data) * self._operator.kmask

    def __call__(self, x):
        if self._real_input:
            x = complex_view_of_real_array(x.reshape(-1, 2))
        if self._flat_input:
            x = x.reshape(self._image_shape)

        diff = self._diff(x)

        return 0.5 * ((diff) * np.conj(diff)).real.sum()

    def gradient(self, x):
        if self._real_input:
            x = complex_view_of_real_array(x.reshape(-1, 2))
        if self._flat_input:
            x = x.reshape(self._image_shape)

        grad = self._operator.adjoint(self._diff(x))

        if self._real_input:
            grad = real_view_of_complex_array(grad)
        if self._flat_input:
            grad = grad.ravel()

        return grad


def signal1d(x, *args):
    return np.sin((2 * np.pi * args[0]) * x) - 2 * np.cos(
        (4 * np.pi * args[0]) * x) + 1j * np.zeros(x.shape)


def signal2d(x, y, *args):
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.sin((4 * np.pi * args[0]) * X) - 2 * np.cos(
        (6 * np.pi * args[0]) * Y) + 2j * np.sin((2 * np.pi * args[0]) * X)


if __name__ == '__main__':
    # number of sample points for high res image (must be square)
    high_res_shape = (128, 128)
    # number of sample points for lower res reconstruction
    low_res_shape = (64, 48)
    kspace_oversampling_factor = 2
    interpolation_size = 6
    sampling = 'cartesian'  # cartesian or radial

    # the frequency of the simulated sine wave
    freq = 1 / (2.5 * np.pi)

    # the location of the sampling points in image space
    x1, dx1 = np.linspace(0,
                          1 / freq,
                          high_res_shape[0],
                          endpoint=False,
                          retstep=True)
    y1, dy1 = np.linspace(0,
                          1 / freq,
                          high_res_shape[1],
                          endpoint=False,
                          retstep=True)

    # high res true image
    f1 = signal2d(x1, y1, freq)

    x2, dx2 = np.linspace(0,
                          1 / freq,
                          low_res_shape[0],
                          endpoint=False,
                          retstep=True)
    y2, dy2 = np.linspace(0,
                          1 / freq,
                          low_res_shape[1],
                          endpoint=False,
                          retstep=True)

    # lower res true image
    f2 = signal2d(x2, y2, freq)

    #-----------------------------------------------------------------------
    # NUFFT that maps from the original grid in image space to the k-space sample points
    # the frequencies for the NUFFT have the scaled to -pi to pi

    # setup the k-space sample points
    if sampling == 'cartesian':
        k = np.linspace(-np.pi, np.pi, high_res_shape[0], endpoint=False)
        k1 = np.zeros((high_res_shape[0]**2, 2))
        k1[:, 0] = np.repeat(k, high_res_shape[0])
        k1[:, 1] = np.tile(k, high_res_shape[0])
    elif sampling == 'radial':
        num_samples_per_spoke = 2 * high_res_shape[0]
        num_spokes = int(high_res_shape[0] * np.pi / 2) // 4
        k = np.linspace(-np.pi, np.pi, num_samples_per_spoke, endpoint=False)
        k1 = np.zeros((num_spokes * num_samples_per_spoke, 2))

        phis = np.linspace(-np.pi, np.pi, num_spokes, endpoint=False)

        for i, phi in enumerate(phis):
            k1[i * num_samples_per_spoke:(i + 1) * num_samples_per_spoke,
               0] = np.cos(phi) * k
            k1[i * num_samples_per_spoke:(i + 1) * num_samples_per_spoke,
               1] = np.sin(phi) * k

    else:
        raise ValueError

    # first NUFFT operator that maps from a high res image grid to kspace points
    nufft1 = NUFFTOperator(
        k1,
        high_res_shape,
        tuple(kspace_oversampling_factor * x for x in high_res_shape),
        interpolation_shape=(interpolation_size, ) * f1.ndim)
    # note: to get the same results as numpy's FFT, we have to fftshift in input to forward
    F1_nu = nufft1.forward(f1)

    # second NUFFT operator that maps from a lower res image grid to kspace points
    # NOTE: to get correct results, we have to scale the k-space frequencies and introduce a scale factor
    k2 = k1.copy()
    for i in range(k2.shape[1]):
        k2[:, i] *= (high_res_shape[i] / low_res_shape[i])

    nufft2 = NUFFTOperator(
        k2,
        low_res_shape,
        tuple(kspace_oversampling_factor * x for x in low_res_shape),
        interpolation_shape=(interpolation_size, ) * f2.ndim,
        scale_factor=(np.array(high_res_shape) /
                      np.array(low_res_shape)).prod())

    F2_nu = nufft2.forward(f2)

    loss_func = DataFidelityLoss(F1_nu,
                                 nufft2,
                                 real_input=True,
                                 flat_input=True,
                                 image_shape=low_res_shape)

    res = fmin_cg(loss_func,
                  np.zeros(2 * f2.size),
                  fprime=loss_func.gradient,
                  maxiter=1000)
    res = complex_view_of_real_array(res.reshape(-1, 2)).reshape(low_res_shape)

    #-----------------------------------------------------------------------
    # plots
    asp = low_res_shape[1] / low_res_shape[0]

    ims = dict(cmap=plt.cm.Greys_r,
               vmax=1.2 * f1.real.max(),
               vmin=1.2 * f1.real.min())

    fig, ax = plt.subplots(2, 3, figsize=(6 * (3 / 2), 6), squeeze=False)
    ax[0, 0].imshow(f1.real, **ims)
    ax[0, 1].imshow(f1.imag, **ims)
    ax[0, 2].imshow(np.abs(f1), **ims)
    ax[1, 0].imshow(res.real, aspect=asp, **ims)
    ax[1, 1].imshow(res.imag, aspect=asp, **ims)
    ax[1, 2].imshow(np.abs(res), aspect=asp, **ims)
    fig.tight_layout()
    fig.show()

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
    ax2.plot(k1[:, 0], k1[:, 1], '.', ms=1)
    fig2.tight_layout()
    fig2.show()
