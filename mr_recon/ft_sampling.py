"""script to understand sampliong of fourier space and discrete FT better"""
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def triangle(x : npt.NDArray) -> npt.NDArray:
    y = np.zeros_like(x)
    ipos = np.where(np.logical_and(x >= 0, x < 1))
    ineg = np.where(np.logical_and(x >= -1, x < 0))

    y[ipos] = 1 - x[ipos]
    y[ineg] = 1 + x[ineg]

    #y[ipos] = 1
    #y[ineg] = 1

    return y

def square(x : npt.NDArray) -> npt.NDArray:
    y = np.zeros_like(x)
    ipos = np.where(np.logical_and(x >= 0, x < 0.5))
    ineg = np.where(np.logical_and(x >= -0.5, x < 0))

    y[ipos] = 1
    y[ineg] = 1

    return y


if __name__ == '__main__':

    n_high = 64
    n_low = 32

    shape = 'square'

    if shape == 'square':
        signal = lambda x: square(x)
        ft_signal = lambda k: np.sinc(k/2/np.pi)
    elif shape == 'triangle':
        signal = lambda x: triangle(x)
        ft_signal = lambda k: np.sinc(k/2/np.pi)**2

    x_high, dx_high = np.linspace(-1,1,n_high, endpoint = False, retstep=True)
    x_low, dx_low = np.linspace(-1,1,n_low, endpoint = False, retstep=True)

    f_high = signal(x_high)
    f_low = signal(x_low)

    F_high = np.fft.fft(f_high)
    F_low = np.fft.fft(f_low)

    k_high = np.fft.fftfreq(x_high.size, d = dx_high) * 2 * np.pi
    k_low = np.fft.fftfreq(x_low.size, d = dx_low) * 2 * np.pi
 
    #In order to get a discretisation of the continuous Fourier transform
    #we need to multiply g by a phase factor
    # see here for details https://stackoverflow.com/questions/24077913/discretized-continuous-fourier-transform-with-numpy
    # note that here we don't include the 1/sqrt(2*pi) factor in the definition of the cont. FT
    F_high_scaled = F_high*(dx_high*np.exp(-1j*k_high*x_high[0]))
    F_low_scaled = F_low*(dx_low*np.exp(-1j*k_low*x_low[0]))

    #---------------------------------------------------------------------
    # plots
    ms = 3

    x_super_high = np.linspace(-1,1,2048*16)
    k_super_high = np.linspace(k_high.min(),k_high.max(),2048*16)

    fig, ax = plt.subplots(1,4, figsize=(4*4,4))
    ax[0].plot(x_super_high, signal(x_super_high), 'k-', lw = 0.5)
    ax[0].plot(x_high, f_high, 'x', ms = ms, color = plt.cm.tab10(0))
    ax[0].plot(x_low, f_low, '.', ms = ms, color = plt.cm.tab10(1))

    ax[1].plot(k_super_high, ft_signal(k_super_high), 'k-', lw = 0.5)
    ax[1].plot(k_high, F_high_scaled.real, 'x', ms = ms, color = plt.cm.tab10(0))
    ax[1].plot(k_low, F_low_scaled.real, '.', ms = ms, color = plt.cm.tab10(1))
    
    ax[2].plot(k_super_high, ft_signal(k_super_high), 'k-', lw = 0.5)
    ax[2].plot(k_high, F_high_scaled.real, 'x', ms = ms, color = plt.cm.tab10(0))
    ax[2].plot(k_low, F_low_scaled.real, '.', ms = ms, color = plt.cm.tab10(1))

    ax[3].plot(k_super_high, ft_signal(k_super_high), 'k-', lw = 0.5)
    ax[3].plot(k_high, F_high_scaled.real, 'x', ms = ms, color = plt.cm.tab10(0))
    ax[3].plot(k_low, F_low_scaled.real, '.', ms = ms, color = plt.cm.tab10(1))

    ax[2].set_xlim(k_low.min(),k_low.max())
    ax[3].set_xlim(k_low.min(),-8)
    tmp = np.abs(ft_signal(k_super_high[k_super_high<=-5])).max()
    ax[3].set_ylim(-1.01*tmp, 1.01*tmp)

    ax[0].set_xlabel("x")
    ax[0].set_title("signal")

    for axx in ax[1:]:
        axx.set_xlabel("k")
        axx.set_title("Re(FT(signal)")

    for axx in ax:
        axx.grid(ls = ":")

    fig.tight_layout()
    fig.show()

