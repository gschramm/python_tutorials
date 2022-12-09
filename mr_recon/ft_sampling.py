"""script to understand sampliong of fourier space and discrete FT better"""
import abc
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class AnalysticalFourierSignal:

    def __init__(self, scale: float = 1.):
        self._scale = scale

    @property
    def scale(self) -> float:
        return self._scale

    @abc.abstractmethod
    def signal(self, x: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    @abc.abstractmethod
    def continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def signal_scaled(self, x: npt.NDArray) -> npt.NDArray:
        return self.signal(x * self.scale)

    def continous_ft_scaled(self, k: npt.NDArray) -> npt.NDArray:
        return self.continous_ft(k / self.scale) / self.scale


class SquareSignal(AnalysticalFourierSignal):

    def signal(self, x: npt.NDArray) -> npt.NDArray:
        y = np.zeros_like(x)
        ipos = np.where(np.logical_and(x >= 0, x < 0.5))
        ineg = np.where(np.logical_and(x >= -0.5, x < 0))

        y[ipos] = 1
        y[ineg] = 1

        return y

    def continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        return np.sinc(k / 2 / np.pi)


class TriangleSignal(AnalysticalFourierSignal):

    def signal(self, x: npt.NDArray) -> npt.NDArray:
        y = np.zeros_like(x)
        ipos = np.where(np.logical_and(x >= 0, x < 1))
        ineg = np.where(np.logical_and(x >= -1, x < 0))

        y[ipos] = 1 - x[ipos]
        y[ineg] = 1 + x[ineg]

        return y

    def continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        return np.sinc(k / 2 / np.pi)**2


class GaussSignal(AnalysticalFourierSignal):

    def signal(self, x: npt.NDArray) -> npt.NDArray:
        return np.exp(-x**2)

    def continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        return np.sqrt(np.pi) * np.exp(-(k**2) / (4))


if __name__ == '__main__':

    n_high = 128
    n_low = 32
    x0 = 1.
    signal = SquareSignal(scale=1.)

    x_high, dx_high = np.linspace(-x0,
                                  x0,
                                  n_high,
                                  endpoint=False,
                                  retstep=True)
    x_low, dx_low = np.linspace(-x0, x0, n_low, endpoint=False, retstep=True)

    f_high = signal.signal_scaled(x_high)
    f_low = signal.signal_scaled(x_low)

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # part 1: check how well the DFT of the discretized signal approximates the cont. FT of the cont. signal
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    F_high = np.fft.fft(f_high)
    F_low = np.fft.fft(f_low)

    k_high = np.fft.fftfreq(x_high.size, d=dx_high) * 2 * np.pi
    k_low = np.fft.fftfreq(x_low.size, d=dx_low) * 2 * np.pi

    #In order to get a discretisation of the continuous Fourier transform
    #we need to multiply g by a phase factor
    # see here for details https://stackoverflow.com/questions/24077913/discretized-continuous-fourier-transform-with-numpy
    # note that here we don't include the 1/sqrt(2*pi) factor in the definition of the cont. FT
    F_high_scaled = F_high * (dx_high * np.exp(-1j * k_high * x_high[0]))
    F_low_scaled = F_low * (dx_low * np.exp(-1j * k_low * x_low[0]))

    ## inverse Fourier transforms
    #iF_low  = np.fft.ifft(F_low)
    #iF_high =  np.fft.ifft(F_high[np.logical_and(k_high <= k_low.max(), k_high >= k_low.min())])

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # part 2: check how inverse DFT of discretely sampled (and truncated) true cont. FT approximates the true signal
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    cont_FT_sampled_high = signal.continous_ft_scaled(k_high)
    cont_FT_sampled_low = signal.continous_ft_scaled(k_low)

    f_recon_high = np.fft.ifft(cont_FT_sampled_high /
                               (dx_high * np.exp(-1j * k_high * x_high[0])))
    f_recon_low = np.fft.ifft(cont_FT_sampled_low /
                              (dx_low * np.exp(-1j * k_low * x_low[0])))

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # Figure 1, visualize how well a DFT of a dicretized signal approximates the continous FT of the continuous signal
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    ms = 4

    x_super_high = np.linspace(-1, 1, 2048 * 16)
    k_super_high = np.linspace(k_high.min(), k_high.max(), 2048 * 16)

    fig, ax = plt.subplots(1, 4, figsize=(4 * 4, 4))
    ax[0].plot(x_super_high, signal.signal_scaled(x_super_high), 'k-', lw=0.5)
    ax[0].plot(x_high, f_high, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[0].plot(x_low, f_low, '.', ms=ms, color=plt.cm.tab10(1))

    ax[1].plot(k_super_high,
               signal.continous_ft_scaled(k_super_high),
               'k-',
               lw=0.5)
    ax[1].plot(k_high, F_high_scaled.real, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[1].plot(k_low, F_low_scaled.real, '.', ms=ms, color=plt.cm.tab10(1))

    ax[2].plot(k_super_high,
               signal.continous_ft_scaled(k_super_high),
               'k-',
               lw=0.5)
    ax[2].plot(k_high, F_high_scaled.real, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[2].plot(k_low, F_low_scaled.real, '.', ms=ms, color=plt.cm.tab10(1))

    ax[3].plot(k_super_high,
               signal.continous_ft_scaled(k_super_high),
               'k-',
               lw=0.5)
    ax[3].plot(k_high, F_high_scaled.real, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[3].plot(k_low, F_low_scaled.real, '.', ms=ms, color=plt.cm.tab10(1))

    ax[2].set_xlim(k_low.min(), k_low.max())
    ax[3].set_xlim(k_low.min(), -8)
    tmp_min = signal.continous_ft_scaled(
        k_super_high[k_super_high <= -5]).min()
    tmp_max = signal.continous_ft_scaled(
        k_super_high[k_super_high <= -5]).max()
    ax[3].set_ylim(tmp_min, tmp_max)

    ax[0].set_xlabel("x")
    ax[0].set_title("signal")

    for axx in ax[1:]:
        axx.set_xlabel("k")

    ax[1].set_title("Re(FT(signal)")
    ax[2].set_title("Re(FT(signal) - zoom 1")
    ax[3].set_title("Re(FT(signal) - zoom 2")

    for axx in ax:
        axx.grid(ls=":")

    fig.tight_layout()
    fig.show()

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # Figure 2: check how inverse DFT of samples from the true cont. FT approximates the true signal
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    x_super_high = np.linspace(-1, 1, 1000)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.plot(x_super_high, signal.signal_scaled(x_super_high), 'k-', lw=0.5)
    ax2.plot(x_high, f_recon_high, 'x-', ms=ms, color=plt.cm.tab10(0), lw=0.5)
    ax2.plot(x_low, f_recon_low, '.-', ms=ms, color=plt.cm.tab10(1), lw=0.5)
    ax2.grid(ls=":")
    ax2.set_xlabel("x")
    ax2.set_title("recon structed signal")
    fig2.tight_layout()
    fig2.show()
