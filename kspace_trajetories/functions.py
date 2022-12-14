import abc
import types
import functools
import numpy as np
import numpy.typing as npt

try:
    import cupy as cp
    import cupy.typing as cpt
except:
    import warnings
    import numpy as np
    import numpy.typing as cpt


class AnalysticalFourierSignal(abc.ABC):
    """abstract base class for 1D signals where the analytical Fourier transform exists"""

    def __init__(self,
                 scale: float = 1.,
                 stretch: float = 1.,
                 shift: float = 0.,
                 xp: types.ModuleType = np,
                 T2star: float = 10.):
        self._scale = scale
        self._stretch = stretch
        self._shift = shift
        self._xp = xp
        self._T2star = T2star

    @property
    def T2star(self) -> float:
        return self._T2star

    @property
    def xp(self) -> types.ModuleType:
        return self._xp

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

    def signal(self, x: npt.NDArray, t: float = 0) -> npt.NDArray:
        return self.xp.exp(-t / self.T2star) * self.scale * self._signal(
            (x - self.shift) * self.stretch)

    def continous_ft(self, k: npt.NDArray, t: float = 0) -> npt.NDArray:
        return self.xp.exp(-t / self.T2star) * self.scale * self.xp.exp(
            -1j * self.shift * k) * self._continous_ft(
                k / self.stretch) / self.xp.abs(self.stretch)


class SquareSignal(AnalysticalFourierSignal):

    def _signal(self, x: npt.NDArray) -> npt.NDArray:
        y = self.xp.zeros_like(x)
        ipos = self.xp.where(self.xp.logical_and(x >= 0, x < 0.5))
        ineg = self.xp.where(self.xp.logical_and(x >= -0.5, x < 0))

        y[ipos] = 1
        y[ineg] = 1

        return y

    def _continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        return self.xp.sinc(k / 2 / self.xp.pi) / self.xp.sqrt(2 * self.xp.pi)


class TriangleSignal(AnalysticalFourierSignal):

    def _signal(self, x: npt.NDArray) -> npt.NDArray:
        y = self.xp.zeros_like(x)
        ipos = self.xp.where(self.xp.logical_and(x >= 0, x < 1))
        ineg = self.xp.where(self.xp.logical_and(x >= -1, x < 0))

        y[ipos] = 1 - x[ipos]
        y[ineg] = 1 + x[ineg]

        return y

    def _continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        return self.xp.sinc(k / 2 / self.xp.pi)**2 / self.xp.sqrt(
            2 * self.xp.pi)


class GaussSignal(AnalysticalFourierSignal):

    def _signal(self, x: npt.NDArray) -> npt.NDArray:
        return self.xp.exp(-x**2)

    def _continous_ft(self, k: npt.NDArray) -> npt.NDArray:
        return self.xp.sqrt(self.xp.pi) * self.xp.exp(
            -(k**2) / (4)) / self.xp.sqrt(2 * self.xp.pi)


class CompoundAnalysticalFourierSignal():

    def __init__(self, signals: list[AnalysticalFourierSignal]):
        self._signals = signals

    @property
    def signals(self) -> list[AnalysticalFourierSignal]:
        return self._signals

    def T2star(self, x: npt.NDArray) -> npt.NDArray:
        return functools.reduce(
            lambda a, b: a + b,
            [z.T2star * (z.signal(x, t=0) > 0) for z in self.signals])

    def signal(self, x: npt.NDArray, t: float = 0) -> npt.NDArray:
        return functools.reduce(lambda a, b: a + b,
                                [z.signal(x, t=t) for z in self.signals])

    def continous_ft(self, x: npt.NDArray, t: float = 0) -> npt.NDArray:
        return functools.reduce(lambda a, b: a + b,
                                [z.continous_ft(x, t=t) for z in self.signals])


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


class Functional(abc.ABC):
    """abstract base class for a functional g(x) =  scale * f(x - shift)"""

    def __init__(self,
                 xp: types.ModuleType,
                 scale: float = 1.,
                 shift: float | npt.NDArray | cpt.NDArray = 0.):
        self._xp = xp
        self._scale = scale  # positive scale factor (g(x) = scale * f(x))
        self._shift = shift

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def shift(self) -> float | npt.NDArray | cpt.NDArray:
        return self._shift

    @property
    def xp(self) -> types.ModuleType:
        return self._xp

    @abc.abstractmethod
    def _call_f(self, x: npt.NDArray | cpt.NDArray) -> float:
        """f(x)"""
        raise NotImplementedError

    def __call__(self, x: npt.NDArray | cpt.NDArray) -> float:
        """g(x) = scale * f(x - shift)"""
        return self._scale * self._call_f(x - self.shift)


class SmoothFunctional(Functional):
    """smooth functional  g(x) =  scale * f(x - shift) where gradient of f(x) is known"""

    @abc.abstractmethod
    def _gradient_f(self,
                    x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """gradient of f(x)"""
        raise NotImplementedError

    def gradient(self,
                 x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """gradient of g(x)"""
        return self.scale * self._gradient_f(x - self.shift)


class ConvexFunctionalWithProx(Functional):
    """abstract class for functional g(x) =  scale * f(x - shift) 
       where either prox_f^sigma or prox_f*^sigma is known (analytically)

       if either of the two proxes is known, we can calculate prox_g^sigma and prox_g*^sigma
       f*/g* are the convex dual functions of g/f
    """

    @abc.abstractmethod
    def prox(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g^sigma"""
        raise NotImplementedError

    @abc.abstractmethod
    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g*^sigma"""
        raise NotImplementedError


class ConvexFunctionalWithPrimalProx(Functional):
    """abstract class for functional g(x) =  scale * f(x - shift) 
       where prox_f^sigma is known (analytically)

       we can calculate prox_g*^sigma using Moreau's identity
       f*/g* are the convex dual functions of g/f
    """

    @abc.abstractmethod
    def _prox_f(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_f^sigma"""
        raise NotImplementedError

    def prox(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g^sigma using pre/post composition rules"""
        return self._prox_f(x - self.shift,
                            sigma=self.scale * sigma) + self.shift

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g*^sigma using Moreau"""
        return x - sigma * self.prox(x / sigma, sigma=1. / sigma)


class ConvexFunctionalWithDualProx(Functional):
    """abstract class for functional g(x) =  scale * f(x - shift) 
       where prox_f*^sigma is known (analytically)

       we can calculate prox_g^sigma using Moreau's identity
       f*/g* are the convex dual functions of g/f
    """

    @abc.abstractmethod
    def _prox_convex_dual_f(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_f*^sigma"""
        raise NotImplementedError

    def prox_convex_dual(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g*^sigma"""
        return self.scale * self._prox_convex_dual_f(
            (x - sigma * self.shift) / self.scale, sigma=sigma / self.scale)

    def prox(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_g^sigma using Moreau"""
        return x - sigma * self.prox_convex_dual(x / sigma, sigma=1. / sigma)


class SquaredL2Norm(SmoothFunctional, ConvexFunctionalWithPrimalProx):
    """squared L2 norm times 0.5"""

    def _call_f(self, x: npt.NDArray | cpt.NDArray) -> float:
        """f(x) = 0.5 * sum_x conj(x_i) * x_i"""
        return 0.5 * (self.xp.conj(x) * x).sum().real

    def _prox_f(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_f^sigma = x / (1 + sigma)"""
        return x / (1 + sigma)

    def _gradient_f(self,
                    x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """gradient f = x"""
        return x


class L2L1Norm(ConvexFunctionalWithDualProx):
    """sum of pointwise Eucliean norms (L2L1 norm)"""

    def _call_f(self, x: npt.NDArray | cpt.NDArray) -> float:
        """f(x) = sum_i SquaredL2Norm(x_i)"""
        if self.xp.isrealobj(x):
            res = self.xp.linalg.norm(x, axis=0).sum()
        else:
            res = self.xp.linalg.norm(
                x.real, axis=0).sum() + self.xp.linalg.norm(x.imag,
                                                            axis=0).sum()

        return res

    def _prox_convex_dual_f(
            self, x: npt.NDArray | cpt.NDArray,
            sigma: float | npt.NDArray | cpt.NDArray
    ) -> npt.NDArray | cpt.NDArray:
        """prox_f*^sigma = projection on L2 balls"""
        if self.xp.isrealobj(x):
            gnorm = self.xp.linalg.norm(x, axis=0)
            r = x / self.xp.clip(gnorm, 1, None)
        else:
            r = self.xp.zeros_like(x)

            gnorm_real = self.xp.linalg.norm(x.real, axis=0)
            r.real = x.real / self.xp.clip(gnorm_real, 1, None)

            gnorm_imag = self.xp.linalg.norm(x.imag, axis=0)
            r.imag = x.imag / self.xp.clip(gnorm_imag, 1, None)

        return r