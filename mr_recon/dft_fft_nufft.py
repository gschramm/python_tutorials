"""demo script to undestand DFT vs FFT vs NUFFT"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pynufft


def signal1d(x, f, type='sine', substract_mean=False):

    if type == 'sine':
        sig = np.sin((2 * np.pi * f) * x) - 2 * np.cos(
            (4 * np.pi * f) * x) + 1j * np.zeros(x.shape)
    elif type == 'gauss':
        sig = np.exp(-15 * f * x**2) - 1j * np.exp(-5 * f * x**2)
    elif type == 'smooth-random':
        tmp0 = gaussian_filter1d(np.random.rand(x.size), 1 / f)
        tmp1 = gaussian_filter1d(np.random.rand(x.size), 1 / f)
        sig = tmp0 + 1j * tmp1
    elif type == 'impulse':
        sig = np.zeros(x.shape, dtype=np.complex128)
        sig[x.size // 2] = 1

    else:
        raise ValueError

    if substract_mean:
        sig -= sig.mean()

    return sig


#----------------------------------------------------------------------------

if __name__ == '__main__':
    n = 32
    np.random.seed(0)

    period = n / 10

    # the location of the sampling points in image space
    x, dx = np.linspace(-period, period, n, endpoint=False, retstep=True)
    f = signal1d(x, 1 / period, type='smooth-random', substract_mean=False)

    #--------------------------------------------------------------------------
    # (1) fast fourier transform
    F_FFT = np.fft.fft(np.fft.fftshift(f))
    # frequencies (omega's) where FFT is calculated
    k_FFT = np.fft.fftfreq(f.size) * (2 * np.pi)

    #--------------------------------------------------------------------------
    # (2) non-uniform FFT at "random" k-space points
    # setup nufft
    k_NUFFT = k_FFT.copy()
    dk_NUFFT = np.sort(k_NUFFT)[1] - np.sort(k_NUFFT)[0]
    k_NUFFT += dk_NUFFT * (np.random.rand(n) - 0.5)

    k_NUFFT = np.clip(k_NUFFT, -np.pi, np.pi)

    nufft = pynufft.NUFFT()
    nufft.plan(np.expand_dims(k_NUFFT, -1), (n, ), (4 * n, ), (18, ))

    F_NUFFT = nufft.forward(f)

    #--------------------------------------------------------------------------
    # (3) manual discrete Fourier transforms (DFTs) on densely sampled k grid
    nn = np.arange(n) - n // 2
    k_DFT = np.linspace(-np.pi, np.pi, 10000)
    F_DFT = np.zeros(k_DFT.shape, dtype=np.complex128)

    for i, k in enumerate(k_DFT):
        F_DFT[i] = (f * np.exp(-1j * k * nn)).sum()

    #--------------------------------------------------------------------------
    # plots

    fig, ax = plt.subplots(2, 2, figsize=(7, 7), sharey='row', sharex='row')
    ax[0, 0].plot(x, f.real, '.-')
    ax[0, 1].plot(x, f.imag, '.-')
    ax[1, 0].plot(k_DFT, F_DFT.real, '-', label='DFT')
    ax[1, 1].plot(k_DFT, F_DFT.imag, '-')
    ax[1, 0].plot(k_FFT, F_FFT.real, 'x', label='FFT')
    ax[1, 1].plot(k_FFT, F_FFT.imag, 'x')
    ax[1, 0].plot(k_NUFFT, F_NUFFT.real, '.', label='NUFFT')
    ax[1, 1].plot(k_NUFFT, F_NUFFT.imag, '.')

    ax[1, 0].legend()

    ax[0, 0].set_xlabel('x')
    ax[0, 1].set_xlabel('x')
    ax[1, 0].set_xlabel('k')
    ax[1, 1].set_xlabel('k')

    ax[0, 0].set_title('Re(signal)')
    ax[0, 1].set_title('Im(signal)')
    ax[1, 0].set_title('Re(DFT(signal))')
    ax[1, 1].set_title('Im(DFT(signal))')

    for axx in ax.ravel():
        axx.grid(ls=':')

    fig.tight_layout()
    fig.show()