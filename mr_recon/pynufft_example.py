import pkg_resources

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

from pynufft import NUFFT

from operators import real_view_of_complex_array


DATA_PATH = pkg_resources.resource_filename('pynufft', './src/data/')
om = np.load(DATA_PATH+'om2D.npz')['arr_0']

om = om[::1,:]

# generate an image x
x = scipy.misc.ascent()[::2, ::2].astype(np.complex64)
x /= x.max()


Nd = (256, 256)  # image size
print('setting image dimension Nd...', Nd)
Kd = (512, 512)  # k-space size
print('setting spectrum dimension Kd...', Kd)
Jd = (6, 6)  # interpolation size
print('setting interpolation size Jd...', Jd)

NufftObj = NUFFT()
NufftObj.plan(om, Nd, Kd, Jd)

#-----------------------------------------------------

# the actual transforms
x_fwd = NufftObj.forward(x)
x_fwd_back = NufftObj.adjoint(x_fwd) # adjoint

# generate a random y data set
y = (np.random.rand(*x_fwd.real.shape) + 1j * np.random.rand(*x_fwd.real.shape)).astype(x.dtype)
y_back = NufftObj.adjoint(y) # adjoint

ip_a = (real_view_of_complex_array(x_fwd)*real_view_of_complex_array(y)).sum()
ip_b = (real_view_of_complex_array(x)*real_view_of_complex_array(y_back)).sum()

print(ip_a, ip_b)


fig, ax = plt.subplots(1,3, figsize = (12,4))
ax[0].imshow(np.abs(x))
ax[1].imshow(np.abs(x_fwd_back))
fig.tight_layout()
fig.show()



