"""script to understand sampliong of fourier space and discrete FT better"""
import numpy as np
import matplotlib.pyplot as plt

from functions import SquareSignal, TriangleSignal, GaussSignal, CompoundAnalysticalFourierSignal, SquaredL2Norm, L2L1Norm
from operators import FFT, GradientOperator
from algorithms import PDHG

if __name__ == '__main__':

    xp = np

    n = 128
    x0 = 110
    noise_level = 0.3
    num_iter = 2000
    beta = 4e1
    prior = 'SquaredL2Norm'

    signal1 = SquareSignal(stretch=1. / x0, scale=1, shift=0)
    signal2 = SquareSignal(stretch=1.5 / x0, scale=-0.5, shift=0)
    #signal3 = TriangleSignal(stretch=8 / x0, scale=0.25, shift=0)
    signal3 = SquareSignal(stretch=8 / x0, scale=0.25, shift=0)
    signal = CompoundAnalysticalFourierSignal([signal1, signal2, signal3])

    x, dx = xp.linspace(-x0, x0, n, endpoint=False, retstep=True)

    fft = FFT(x)
    k = fft.k

    if prior == 'SquaredL2Norm':
        prior_norm = SquaredL2Norm(xp, scale=beta)
    elif prior == 'L1L2Norm':
        prior_norm = L2L1Norm(xp, scale=beta)
    else:
        raise ValueError
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # generate data from continuous FFT
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    noise_free_data = signal.continous_ft(k)

    data = noise_free_data.copy() + noise_level * xp.random.randn(
        *noise_free_data.shape) + 1j * noise_level * xp.random.randn(
            *noise_free_data.shape)

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # inverse fourier transform reconstruction
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    recon1 = fft.inverse(data)

    data_distance = SquaredL2Norm(xp, scale=1.0, shift=data)

    prior_operator = GradientOperator(x.shape, xp=xp)

    fft_norm = fft.norm(num_iter=200)

    pdhg = PDHG(data_operator=fft,
                data_distance=data_distance,
                sigma=1. / fft_norm,
                tau=1. / fft_norm,
                prior_operator=prior_operator,
                prior_functional=prior_norm)
    pdhg.run(num_iter, verbose=False, calculate_cost=True)

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # plots
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    xx = xp.linspace(-x0, x0, 1000, endpoint=False)
    kk = xp.linspace(k.min(), k.max(), 1000, endpoint=False)
    it = np.arange(1, num_iter + 1)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].plot(xx, signal.signal(xx).real, 'k-', lw=0.5)
    ax[0].plot(x, recon1.real, '.-', lw=0.3)
    ax[0].plot(x, pdhg.x.real, '.-', lw=0.3)
    ax[1].plot(xx, signal.signal(xx).imag, 'k-', lw=0.5)
    ax[1].plot(x, recon1.imag, '.-', lw=0.3)
    ax[1].plot(x, pdhg.x.imag, '.-', lw=0.3)
    ax[2].plot(kk, signal.continous_ft(kk).real, 'k-', lw=0.5)
    ax[2].plot(k, noise_free_data.real, 'x', ms=4)
    ax[2].plot(k, data.real, '.', ms=4)
    ax[3].plot(it, pdhg.cost)
    ax[3].set_ylim(None, pdhg.cost[20:].max())

    for axx in ax.ravel():
        axx.grid(ls=':')
    ax[1].set_ylim(*ax[0].get_ylim())

    fig.tight_layout()
    fig.show()