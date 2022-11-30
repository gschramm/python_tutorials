import numpy as np
from scipy.optimize import fmin_cg
from scipy.misc import face
import matplotlib.pyplot as plt

import pynufft

from operators import real_view_of_complex_array, complex_view_of_real_array
from phantoms import rod_phantom


class FwdModel:

    def __init__(self, nufft, flat_input=True, pseudo_complex_mode=True):
        self._nufft = nufft
        self._image_shape = nufft.Nd
        self._flatten_input = flat_input
        self._pseudo_complex_mode = pseudo_complex_mode

    def forward(self, x):
        input_dtype = x.dtype

        if self._flatten_input:
            if self._pseudo_complex_mode:
                x = complex_view_of_real_array(
                    x.reshape(self._image_shape + (2, )))
            else:
                x = x.reshape(self._image_shape)
        else:
            if self._pseudo_complex_mode:
                x = complex_view_of_real_array(x)

        x_fwd = self._nufft.forward(x)

        if self._pseudo_complex_mode:
            x_fwd = real_view_of_complex_array(x_fwd).astype(input_dtype)

        return x_fwd

    def adjoint(self, y):
        input_dtype = y.dtype

        if self._pseudo_complex_mode:
            y = complex_view_of_real_array(y)

        y_back = self._nufft.adjoint(y) * self._nufft.Kdprod

        if self._flatten_input:
            if self._pseudo_complex_mode:
                y_back = real_view_of_complex_array(y_back).ravel()
            else:
                y_back = y_back.ravel()
        else:
            if self._pseudo_complex_mode:
                y_back = real_view_of_complex_array(y_back)

        if self._pseudo_complex_mode:
            y_back = y_back.astype(input_dtype)

        return y_back


class DataFidelity:

    def __init__(self, fwd_model, data):
        self._fwd_model = fwd_model
        self._data = data

    def __call__(self, x):
        exp_data = self._fwd_model.forward(x)

        return 0.5 * ((exp_data - self._data)**2).sum()

    def gradient(self, x):
        exp_data = self._fwd_model.forward(x)
        return self._fwd_model.adjoint(exp_data - self._data)


if __name__ == '__main__':
    n = 512
    num_iterations = 50

    #num_spokes = int(n * np.pi / 2)
    num_spokes = 20
    num_samples_per_spoke = 2 * n

    om = np.zeros((num_spokes, num_samples_per_spoke, 2))
    k = np.linspace(-np.pi, np.pi, num_samples_per_spoke)
    spoke_angles = (np.arange(num_spokes) * (2 * np.pi /
                                             (1 + np.sqrt(5)))) % (2 * np.pi)

    for i, spoke_angle in enumerate(spoke_angles):
        om[i, :, 0] = k * np.cos(spoke_angle)
        om[i, :, 1] = k * np.sin(spoke_angle)

    om = om.reshape((num_samples_per_spoke * num_spokes, 2))

    Nd = (n, n)  # image size
    print('setting image dimension Nd...', Nd)
    Kd = (2 * n, 2 * n)  # k-space size
    print('setting spectrum dimension Kd...', Kd)
    Jd = (6, 6)  # interpolation size
    print('setting interpolation size Jd...', Jd)

    nufftObj = pynufft.NUFFT()
    nufftObj.plan(om, Nd, Kd, Jd)

    model = FwdModel(nufftObj, flat_input=True, pseudo_complex_mode=True)

    # generate pseudo-complex image
    x = face()[-n:, -n:, 0].astype(np.float32)
    x = np.stack([x, np.zeros_like(x)], axis=-1).ravel()

    #-----------------------------------------------------

    # the actual transforms
    x_fwd = model.forward(x)
    x_fwd_back = model.adjoint(x_fwd)

    # generate a random y data set
    y = np.random.rand(*x_fwd.real.shape)
    y_back = model.adjoint(y)

    ip_a = (x_fwd * y).sum()
    ip_b = (x * y_back).sum()

    print(ip_a, ip_b, ip_a / ip_b)

    data_fidelity_loss = DataFidelity(model, x_fwd)

    # cg recon method 1
    recon_cg = nufftObj.solve(complex_view_of_real_array(x_fwd),
                              'cg',
                              maxiter=num_iterations)

    # cg recon method 2
    recon_cg_2 = fmin_cg(data_fidelity_loss,
                         0 * x,
                         fprime=data_fidelity_loss.gradient,
                         maxiter=num_iterations)

    ims = dict(cmap=plt.cm.Greys_r)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(np.linalg.norm(x.reshape(n, n, 2), axis=-1), **ims)
    ax[1].imshow(np.abs(recon_cg), **ims)
    ax[2].imshow(np.linalg.norm(recon_cg_2.reshape(n, n, 2), axis=-1), **ims)
    fig.tight_layout()
    fig.show()
