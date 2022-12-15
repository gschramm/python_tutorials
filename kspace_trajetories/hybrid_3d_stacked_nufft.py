import numpy as np
import pynufft
from functions import SquaredL2Norm
from operators import FFT, MultiChannelNonCartesianMRAcquisitionModel
import matplotlib.pyplot as plt


def fake_Cartesian(Nd):
    dim = len(Nd)  # dimension
    M = np.prod(Nd)
    om = np.zeros((M, dim), dtype=float)
    grid = np.indices(Nd)
    for dimid in range(0, dim):
        om[:, dimid] = (grid[dimid].ravel() * 2 / Nd[dimid] - 1.0) * np.pi
    return om


np.random.seed(1)
x0 = 110
n = 8
eps = 1e-3

# setup a signal and simulate data via the forward model
f = np.random.rand(n, n, n) + 1j * np.random.rand(n, n, n)
f_fwd1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f)))

#--------------
# full 3D nufft
#--------------
nufft = pynufft.NUFFT(pynufft.helper.device_list()[0])
nufft.plan(fake_Cartesian((n, n, n)), f.shape, tuple([4 * x for x in f.shape]),
           f.ndim * (12, ))
f_fwd2 = nufft.forward(f).reshape(f_fwd1.shape)

#--------------
# hybrid FFT - 2D NUFFT
#--------------

# first perform a 1D FFT along the 0 axis
tmp = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(f, axes=0), axes=[0]),
                      axes=0)

# then perform a series of 2D NUFFTs
nufft_2d = pynufft.NUFFT(pynufft.helper.device_list()[0])
nufft_2d.plan(fake_Cartesian((n, n)), (n, n), (4 * n, 4 * n), (12, 12))

f_fwd3 = np.zeros_like(f_fwd2)
for i in range(n):
    f_fwd3[i, ...] = nufft_2d.forward(tmp[i, ...]).reshape(n, n)
