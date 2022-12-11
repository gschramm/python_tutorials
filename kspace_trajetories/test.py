"""script to understand sampliong of fourier space and discrete FT better"""
import numpy as np
import matplotlib.pyplot as plt

from functions import SquareSignal, TriangleSignal, GaussSignal, CompoundAnalysticalFourierSignal, SquaredL2Norm
from operators import FFT, GradientOperator
from algorithms import PDHG

if __name__ == '__main__':

    xp = np

    n = 128
    x0 = 110
    signal1 = SquareSignal(stretch=1. / x0, scale=1, shift=0)
    signal2 = SquareSignal(stretch=1.5 / x0, scale=-0.5, shift=0)
    signal3 = TriangleSignal(stretch=8 / x0, scale=0.25, shift=0)
    signal = CompoundAnalysticalFourierSignal([signal1, signal2, signal3])

    x, dx = xp.linspace(-x0, x0, n, endpoint=False, retstep=True)

    fft = FFT(x)
    k = fft.k

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # generate data from continuous FFT
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    data = signal.continous_ft(k)

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # inverse fourier transform reconstruction
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    recon1 = fft.inverse(data)

    data_distance = SquaredL2Norm(xp, scale=1.0, shift=data)

    prior_operator = GradientOperator(x.shape, xp=xp)
    prior_norm = SquaredL2Norm(xp, scale=1e1)

    fft_norm = fft.norm(num_iter=200)

    pdhg = PDHG(data_operator=fft,
                data_distance=data_distance,
                sigma=1. / fft_norm,
                tau=1. / fft_norm,
                prior_operator=prior_operator,
                prior_functional=prior_norm)
    pdhg.run(1000, verbose=False, calculate_cost=True)

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # plots
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    xx = xp.linspace(-x0, x0, 1000, endpoint=False)
    kk = xp.linspace(k.min(), k.max(), 1000, endpoint=False)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(xx, signal.signal(xx).real, 'k-', lw=0.5)
    ax[0].plot(x, recon1.real, '.-', lw=0.3)
    ax[0].plot(x, pdhg.x.real, '.-', lw=0.3)
    ax[1].plot(kk, signal.continous_ft(kk).real, 'k-', lw=0.5)
    ax[1].plot(k, data.real, '.')
    fig.tight_layout()
    fig.show()