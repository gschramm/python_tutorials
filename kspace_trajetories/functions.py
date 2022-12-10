import abc
import functools
import numpy as np
import numpy.typing as npt


class AnalysticalFourierSignal(abc.ABC):
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
