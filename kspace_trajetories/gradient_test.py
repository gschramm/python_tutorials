import numpy as np
from functions import SquaredL2Norm
from operators import FFT, MultiChannelNonCartesianMRAcquisitionModel

np.random.seed(1)
x0 = 110
n = 64
eps = 1e-3
mode = 'NUFFT'

# setup a linear test operator that maps from C -> C
x, dx = np.linspace(-x0, x0, n, endpoint=False, retstep=True)

if mode == 'FFT':
    data_operator = FFT(x, xp=np, dtype=np.complex128)
elif mode == 'NUFFT':
    sens = np.ones((1, n), dtype=np.complex128)
    kspace_points = 2 * np.pi * np.random.rand(64, 1) - np.pi
    data_operator = MultiChannelNonCartesianMRAcquisitionModel(
        x.shape, sens, kspace_points, scaling_factor=1 / 18.)
else:
    raise ValueError

# setup a signal and simulate data via the forward model
f = np.random.rand(n) + 1j * np.random.rand(n)
data = data_operator.forward(f)
data_distance = SquaredL2Norm(np, scale=1.0, shift=data)

# set a signal at which we want to test the gradient
f0 = np.random.rand(n) + 1j * np.random.rand(n)
grad = data_operator.adjoint(data_distance.gradient(data_operator.forward(f0)))
l0 = data_distance(data_operator.forward(f0))

kws = {'rtol': 3e-4}

abs_real_err = np.zeros(n)
abs_imag_err = np.zeros(n)

rel_real_err = np.zeros(n)
rel_imag_err = np.zeros(n)

# numerically approximate real part of the gradient
for i in range(n):
    f1 = f0.copy()
    f1[i] += eps
    l1 = data_distance(data_operator.forward(f1))
    g_real_numerical = (l1 - l0) / eps

    abs_real_err[i] = np.abs(g_real_numerical - grad[i].real)
    rel_real_err[i] = abs_real_err[i] / np.abs(grad[i].real)
    #assert (np.isclose(g_real_numerical, grad[i].real, **kws))

    # numerically approximate imaginary part of the gradient
    f2 = f0.copy()
    f2[i] += eps * (1j)
    l2 = data_distance(data_operator.forward(f2))
    g_imag_numerical = (l2 - l0) / eps

    abs_imag_err[i] = np.abs(g_real_numerical - grad[i].real)
    rel_imag_err[i] = abs_imag_err[i] / np.abs(grad[i].imag)
    #assert (np.isclose((l2 - l0) / eps, grad[i].imag, **kws))

print(f'eps {eps:.1e}')
print(f'max relative real error {rel_real_err.max():.2e}')
print(f'max relative imag error {rel_imag_err.max():.2e}')