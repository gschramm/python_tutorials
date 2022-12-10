"""script to understand sampliong of fourier space and discrete FT better"""
import abc
import functools
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class AnalysticalFourierSignal:
    """abstract base class for 1D signals where the analytical Fourier transform exists"""

    def __init__(self,
                 scale: float = 1.,
                 stretch: float = 1.,
                 shift: float = 0.):
        self._scale = scale
        self._stretch = stretch
        self._shift = shift

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def stretch(self) -> float:
        return self._stretch

    @property
    def shift(self) -> float:
        return self._shift

    @abc.abstractmethod
    def _signal(self, x: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    @abc.abstractmethod
    def _continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def signal(self, x: npt.NDArray) -> npt.NDArray:
        return self.scale * self._signal((x - self.shift) * self.stretch)

    def continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        return self.scale * np.exp(-1j * self.shift * k) * self._continous_ft(
            k / self.stretch) / np.abs(self.stretch)


class SquareSignal(AnalysticalFourierSignal):

    def _signal(self, x: npt.NDArray) -> npt.NDArray:
        y = np.zeros_like(x)
        ipos = np.where(np.logical_and(x >= 0, x < 0.5))
        ineg = np.where(np.logical_and(x >= -0.5, x < 0))

        y[ipos] = 1
        y[ineg] = 1

        return y

    def _continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        return np.sinc(k / 2 / np.pi) / np.sqrt(2 * np.pi)


class TriangleSignal(AnalysticalFourierSignal):

    def _signal(self, x: npt.NDArray) -> npt.NDArray:
        y = np.zeros_like(x)
        ipos = np.where(np.logical_and(x >= 0, x < 1))
        ineg = np.where(np.logical_and(x >= -1, x < 0))

        y[ipos] = 1 - x[ipos]
        y[ineg] = 1 + x[ineg]

        return y

    def _continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        return np.sinc(k / 2 / np.pi)**2 / np.sqrt(2 * np.pi)


class GaussSignal(AnalysticalFourierSignal):

    def _signal(self, x: npt.NDArray) -> npt.NDArray:
        return np.exp(-x**2)

    def _continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        return np.sqrt(np.pi) * np.exp(-(k**2) / (4)) / np.sqrt(2 * np.pi)


class CompoundAnalysticalFourierSignal():

    def __init__(self, signals: list[AnalysticalFourierSignal]):
        self._signals = signals

    @property
    def signals(self) -> list[AnalysticalFourierSignal]:
        return self._signals

    def signal(self, x: npt.NDArray) -> npt.NDArray:
        return functools.reduce(lambda a, b: a + b,
                                [z.signal(x) for z in self.signals])

    def continous_ft(self, x: npt.NDArray) -> npt.NDArray:
        return functools.reduce(lambda a, b: a + b,
                                [z.continous_ft(x) for z in self.signals])


class FFT:

    def __init__(self, x: npt.NDArray) -> None:
        self._dx = x[1] - x[0]
        self._x = x
        self._phase_factor = self.dx * np.exp(-1j * self.k * x[0])
        self._scale_factor = np.sqrt(x.size / (2 * np.pi))
        self._adjoint_factor = 1 / ((x.size / 2)**2)

    @property
    def x(self) -> npt.NDArray:
        return self._x

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def k(self) -> npt.NDArray:
        return np.fft.fftfreq(self.x.size, d=self.dx) * 2 * np.pi

    @property
    def k_scaled(self) -> npt.NDArray:
        return self.k * self.dx

    @property
    def phase_factor(self) -> npt.NDArray:
        return self._phase_factor

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @property
    def adjoint_factor(self) -> float:
        return self._adjoint_factor

    def forward(self, f: npt.NDArray) -> npt.NDArray:
        return np.fft.fft(f,
                          norm='ortho') * self.phase_factor * self.scale_factor

    def adjoint(self, ft: npt.NDArray) -> npt.NDArray:
        return np.fft.ifft(ft * self.scale_factor / self.phase_factor,
                           norm='ortho') * self._adjoint_factor

    def inverse(self, ft: npt.NDArray) -> npt.NDArray:
        return self.adjoint(ft) / (self.scale_factor**2) / self.adjoint_factor


#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    n_high = 128
    n_low = 64
    x0 = 1.0
    signal1 = SquareSignal(stretch=1., scale=1, shift=0.0)
    signal2 = SquareSignal(stretch=1.5, scale=-0.5, shift=0.0)
    signal3 = SquareSignal(stretch=4, scale=0.25, shift=0.0)
    signal = CompoundAnalysticalFourierSignal([signal1, signal2, signal3])

    x_high, dx_high = np.linspace(-x0,
                                  x0,
                                  n_high,
                                  endpoint=False,
                                  retstep=True)
    x_low, dx_low = np.linspace(-x0, x0, n_low, endpoint=False, retstep=True)

    f_high = signal.signal(x_high)
    f_low = signal.signal(x_low)

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # part 1: check how well the DFT of the discretized signal approximates the cont. FT of the cont. signal
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    fft_high_res = FFT(x_high)
    fft_low_res = FFT(x_low)

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
    # part 3: check adjointness of operators
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    np.random.seed(1)
    model = fft_high_res
    n = n_high

    x = np.random.rand(n) + 1j * np.random.rand(n)
    y = np.random.rand(n) + 1j * np.random.rand(n)

    x_fwd = model.forward(x)
    y_adj = model.adjoint(y)

    ip1 = (np.conj(y) * x_fwd).sum()
    ip2 = (np.conj(y_adj) * x).sum()

    assert (np.isclose(ip1, ip2))

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # Figure 1, visualize how well a DFT of a dicretized signal approximates the continous FT of the continuous signal
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    ms = 4

    x_super_high = np.linspace(-1, 1, 2048 * 16)
    k_super_high = np.linspace(1.2 * k_high.min(), 1.2 * k_high.max(),
                               2048 * 16)

    fig, ax = plt.subplots(3, 3, figsize=(14, 9), sharey='row')
    ax[0, 0].plot(x_super_high, signal.signal(x_super_high), 'k-', lw=0.5)
    ax[0, 0].plot(x_high, f_high, 'x', ms=ms, color=plt.cm.tab10(0))
    ax[0, 0].plot(x_low, f_low, '.', ms=ms, color=plt.cm.tab10(1))

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
    ax[1, 2].set_xlim(k_low.min(), -8)

    ax[2, 1].set_xlim(k_low.min(), k_low.max())
    ax[2, 2].set_xlim(k_low.min(), -8)

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