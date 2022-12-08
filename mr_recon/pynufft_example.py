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
    n = 1024
    num_iterations = 100

    num_spokes = int(n * np.pi / 2) // 32
    #num_spokes = 20
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

    nufftObj = pynufft.NUFFT(pynufft.helper.device_list()[0])
    nufftObj.plan(om, Nd, Kd, Jd)

    model = FwdModel(nufftObj, flat_input=True, pseudo_complex_mode=True)

    # generate pseudo-complex image
    x = np.fromfile('xcat_slice_sag.v',
                    dtype=np.float32).reshape(1024,
                                              1024)[:n, :n].astype(np.float32)
    x = np.stack([x, np.zeros_like(x)], axis=-1)

    #-----------------------------------------------------

    # the actual transforms
    x_fwd = model.forward(x)
    x_fwd_back = model.adjoint(x_fwd)

    # generate a random y data set
    y = np.random.rand(*x_fwd.real.shape)
    y_back = model.adjoint(y)

    ip_a = (x_fwd * y).sum()
    ip_b = (x.ravel() * y_back).sum()

    print(ip_a, ip_b, ip_a / ip_b)

    data_fidelity_loss = DataFidelity(model, x_fwd)

    # cg recon method
    recon_cg = fmin_cg(data_fidelity_loss,
                       np.zeros(x.size),
                       fprime=data_fidelity_loss.gradient,
                       maxiter=num_iterations)

    # reshape the flattened cg recon
    recon_cg = recon_cg.reshape(n, n, 2)

    ims = dict(cmap=plt.cm.Greys_r)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].imshow(np.linalg.norm(x, axis=-1)[::-1, :], **ims)
    ax[1].imshow(np.linalg.norm(recon_cg, axis=-1)[::-1, :], **ims)
    fig.tight_layout()
    fig.show()
