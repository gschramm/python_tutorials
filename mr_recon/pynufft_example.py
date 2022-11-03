import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import pynufft

from operators import real_view_of_complex_array

np.random.seed(1)
num_k = int((256**2))

# generate random k-space sampling points
om = np.random.randn(num_k, 2)

# generate an image x
x = scipy.misc.ascent()[::2, ::2].astype(np.complex64)
x /= x.max()


Nd = (256, 256)  # image size
print('setting image dimension Nd...', Nd)
Kd = (512, 512)  # k-space size
print('setting spectrum dimension Kd...', Kd)
Jd = (6, 6)  # interpolation size
print('setting interpolation size Jd...', Jd)

nufftObj = pynufft.NUFFT()
nufftObj.plan(om, Nd, Kd, Jd)

#-----------------------------------------------------
adj_scaling_factor = np.prod(Kd)

# the actual transforms
x_fwd = nufftObj.forward(x)
x_fwd_back = nufftObj.adjoint(x_fwd) * adj_scaling_factor # adjoint

# generate a random y data set
y = (np.random.rand(*x_fwd.real.shape) + 1j * np.random.rand(*x_fwd.real.shape)).astype(x.dtype)
y_back = nufftObj.adjoint(y) * adj_scaling_factor # adjoint

ip_a = (real_view_of_complex_array(x_fwd)*real_view_of_complex_array(y)).sum()
ip_b = (real_view_of_complex_array(x)*real_view_of_complex_array(y_back)).sum()

print(ip_a, ip_b, ip_a/ip_b)

recon_cg = nufftObj.solve(x_fwd,'cg', maxiter=50)

fig, ax = plt.subplots(1,3, figsize = (12,4))
ax[0].imshow(np.abs(x))
ax[1].imshow(np.abs(x_fwd_back))
ax[2].imshow(np.abs(recon_cg))

fig.tight_layout()
fig.show()