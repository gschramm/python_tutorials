"""script to understand sampliong of fourier space and discrete FT better"""
import numpy as np
import matplotlib.pyplot as plt

from functions import SquareSignal, TriangleSignal, GaussSignal, CompoundAnalysticalFourierSignal
from operators import FFT

if __name__ == '__main__':

    n_high = 128
    n_low = 64
    x0 = 110
    signal1 = SquareSignal(stretch=1. / x0, scale=1, shift=0)
    signal2 = SquareSignal(stretch=1.5 / x0, scale=-0.5, shift=0)
    signal3 = TriangleSignal(stretch=8 / x0, scale=0.25, shift=0)
    signal = CompoundAnalysticalFourierSignal([signal1, signal2, signal3])

    x_high, dx_high = np.linspace(-x0,
                                  x0,
                                  n_high,
                                  endpoint=False,
                                  retstep=True)
    x_low, dx_low = np.linspace(-x0, x0, n_low, endpoint=False, retstep=True)

    f_high = signal.signal(x_high).astype(complex)
    f_low = signal.signal(x_low).astype(complex)

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # part 1: check how well the DFT of the discretized signal approximates the cont. FT of the cont. signal
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    fft_high_res = FFT(x_high)
    fft_low_res = FFT(x_low)

    fft_high_res.adjointness_test()
    fft_low_res.adjointness_test()

    k_high = fft_high_res.k
    k_low = fft_low_res.k

    F_high = fft_high_res.forward(f_high)
    F_low = fft_low_res.forward(f_low)

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # part 2: check how inverse DFT of discretely sampled (and truncated) true cont. FT approximates the true signal
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    cont_FT_sampled_high = signal.continous_ft(k_high)
    cont_FT_sampled_low = signal.continous_ft(k_low)

    f_recon_high = fft_high_res.inverse(cont_FT_sampled_high)
    f_recon_low = fft_low_res.inverse(cont_FT_sampled_low)

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # Figure 1, visualize how well a DFT of a dicretized signal approximates the continous FT of the continuous signal
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    ms = 4

    x_super_high = np.linspace(-x0, x0, 2048 * 16)
    k_super_high = np.linspace(1.2 * k_high.min(), 1.2 * k_high.max(),
                               2048 * 16)

    fig, ax = plt.subplots(3, 3, figsize=(14, 9), sharey='row')
    ax[0, 0].plot(x_super_high, signal.signal(x_super_high), 'k-', lw=0.5)
    ax[0, 0].plot(x_high, f_high.real, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[0, 0].plot(x_low, f_low.real, '.', ms=ms, color=plt.cm.tab10(1))

    ax[0, 1].plot(x_super_high, signal.signal(x_super_high), 'k-', lw=0.5)
    ax[0, 1].plot(x_high,
                  f_recon_high.real,
                  'x-',
                  ms=ms,
                  color=plt.cm.tab10(0),
                  lw=0.5)
    ax[0, 1].plot(x_low,
                  f_recon_low.real,
                  '.-',
                  ms=ms,
                  color=plt.cm.tab10(1),
                  lw=0.5)

    ax[0, 2].plot(x_high,
                  f_recon_high.imag,
                  'x-',
                  ms=ms,
                  color=plt.cm.tab10(0),
                  lw=0.5)
    ax[0, 2].plot(x_low,
                  f_recon_low.imag,
                  '.-',
                  ms=ms,
                  color=plt.cm.tab10(1),
                  lw=0.5)

    for k in range(3):
        ax[1, k].plot(k_super_high,
                      signal.continous_ft(k_super_high).real,
                      'k-',
                      lw=0.5)
    ax[1, 0].plot(k_high, F_high.real, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[1, 0].plot(k_low, F_low.real, '.', ms=ms, color=plt.cm.tab10(1))

    ax[1, 1].plot(k_high, F_high.real, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[1, 1].plot(k_low, F_low.real, '.', ms=ms, color=plt.cm.tab10(1))

    ax[1, 2].plot(k_high, F_high.real, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[1, 2].plot(k_low, F_low.real, '.', ms=ms, color=plt.cm.tab10(1))

    for k in range(3):
        ax[2, k].plot(k_super_high,
                      signal.continous_ft(k_super_high).imag,
                      'k-',
                      lw=0.5)
    ax[2, 0].plot(k_high, F_high.imag, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[2, 0].plot(k_low, F_low.imag, '.', ms=ms, color=plt.cm.tab10(1))

    ax[2, 0].plot(k_super_high,
                  signal.continous_ft(k_super_high).imag,
                  'k-',
                  lw=0.5)
    ax[2, 1].plot(k_high, F_high.imag, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[2, 1].plot(k_low, F_low.imag, '.', ms=ms, color=plt.cm.tab10(1))
    ax[2, 2].plot(k_high, F_high.imag, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[2, 2].plot(k_low, F_low.imag, '.', ms=ms, color=plt.cm.tab10(1))

    ax[1, 1].set_xlim(k_low.min(), k_low.max())
    ax[1, 2].set_xlim(k_low.min(), k_low.min() / 6)

    ax[2, 1].set_xlim(k_low.min(), k_low.max())
    ax[2, 2].set_xlim(k_low.min(), k_low.min() / 6)

    for k in range(3):
        ax[0, k].set_xlabel("x")
    ax[0, 0].set_title("signal", fontsize='medium')
    ax[0, 1].set_title("recon of sampled cont. FT (Re)", fontsize='medium')
    ax[0, 2].set_title("recon of sampled cont. FT (Im)", fontsize='medium')

    for axx in ax[1:, :].ravel():
        axx.set_xlabel("k")

    ax[1, 0].set_title("Re(FT(signal)", fontsize='medium')
    ax[1, 1].set_title("Re(FT(signal) - zoom 1", fontsize='medium')
    ax[1, 2].set_title("Re(FT(signal) - zoom 2", fontsize='medium')

    ax[2, 0].set_title("Im(FT(signal)", fontsize='medium')
    ax[2, 1].set_title("Im(FT(signal) - zoom 1", fontsize='medium')
    ax[2, 2].set_title("Im(FT(signal) - zoom 2", fontsize='medium')

    for axx in ax.ravel():
        axx.grid(ls=":")

    fig.tight_layout()
    fig.show()