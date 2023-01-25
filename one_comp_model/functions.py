import abc
import functools
import numpy as np
import numpy.typing as npt

class Function(abc.ABC):
    def __init__(self) -> None:
        self._scale = 1.

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value) -> None:
        self._scale = value

    @abc.abstractmethod
    def _call(self, t: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def __call__(self, t: float | npt.NDArray) -> float | npt.NDArray:
        if not isinstance(t,np.ndarray):
            t = np.array([t])
        
        res = self.scale * self._call(t)
        if isinstance(res,np.ndarray) and (res.shape[0] == 1):
            res = res[0]

        return res


class IntegrableFunction(Function):
    """function with known indefinite integral"""

    @property
    @abc.abstractmethod
    def _indefinite_integral(self) -> Function:
        raise NotImplementedError

    @property
    def indefinite_integral(self) -> Function:
        f = self._indefinite_integral
        f.scale *= self.scale
        return f

class ExpConvFunction(IntegrableFunction):
    """function where convolution with exp(-kt) kernel is known"""

    @abc.abstractmethod
    def _expconv(self, k2: float) -> IntegrableFunction:
        raise NotImplementedError

    def expconv(self, k2: float) -> IntegrableFunction:
        f = self._expconv(k2)
        f.scale *= self.scale
        return f

class SumFunction(Function):
    """sum of functions"""
    def __init__(self, funcs: list[Function]):
        self._funcs = funcs
        super().__init__()

    @property
    def funcs(self) -> list[Function]:
        return self._funcs

    def _call(self, t: npt.NDArray) -> npt.NDArray:
        return functools.reduce(lambda a, b: a + b,
                                [z(t) for z in self.funcs])


class IntegrableSumFunction(IntegrableFunction):
    """sum of integrable functions"""
    def __init__(self, funcs: list[IntegrableFunction]):
        self._funcs = funcs
        super().__init__()

    @property
    def funcs(self) -> list[IntegrableFunction]:
        return self._funcs

    def _call(self, t: npt.NDArray) -> npt.NDArray:
        return functools.reduce(lambda a, b: a + b,
                                [z(t) for z in self.funcs])
    @property
    def _indefinite_integral(self) -> SumFunction:
        return SumFunction([z.indefinite_integral for z in self._funcs])


class ExpConvSumFunction(ExpConvFunction):
    """sum of integrable functions"""
    def __init__(self, funcs: list[ExpConvFunction]):
        self._funcs = funcs
        super().__init__()

    @property
    def funcs(self) -> list[ExpConvFunction]:
        return self._funcs

    def _call(self, t: npt.NDArray) -> npt.NDArray:
        return functools.reduce(lambda a, b: a + b,
                                [z(t) for z in self.funcs])
    @property
    def _indefinite_integral(self) -> IntegrableSumFunction:
        return IntegrableSumFunction([z.indefinite_integral for z in self._funcs])

    def _expconv(self, k2: float) -> IntegrableSumFunction:
        return IntegrableSumFunction([z.expconv(k2) for z in self._funcs])

#---------------------------------------------------------------------------

class PowerFunction(IntegrableFunction):
    def __init__(self, power: float) -> None:
        if power == -1:
            raise ValueError

        self._power = power
        super().__init__()
    
    @property
    def power(self) -> float:
        return self._power

    def _call(self, t: npt.NDArray) -> npt.NDArray:
        return t**self.power

    @property
    def _indefinite_integral(self) -> Function:
        f = PowerFunction(self.power + 1)
        f.scale = 1/(self.power + 1)
        return f


class ExpDecayFunction(ExpConvFunction):
    def __init__(self, alpha : float):
        if alpha < 0:
            raise ValueError
        self._alpha = alpha
        super().__init__()

    @property
    def alpha(self) -> float:
        return self._alpha

    def _call(self, t: npt.NDArray):
        return np.exp(-self.alpha*t)

    @property
    def _indefinite_integral(self) -> Function:
        f = ExpDecayFunction(self.alpha)
        f.scale =  -1/self.alpha
        return f

    def _expconv(self, k2: float) -> IntegrableFunction:
        """convolution of function with exp(-k2*t) kernel"""

        if k2 == self.alpha:
            raise ValueError

        s = 1 / (k2 - self.alpha)
        f1 = ExpDecayFunction(self.alpha)
        f1.scale = s

        f2 = ExpDecayFunction(k2)
        f2.scale = -s

        return IntegrableSumFunction([f1, f2])


class PlateauFunction(ExpConvFunction):
    def _call(self, t: npt.NDArray):
        return np.ones(t.size)

    @property
    def _indefinite_integral(self) -> Function:
        return PowerFunction(1)

    def _expconv(self, k2: float) -> IntegrableFunction:
        """convolution of function with exp(-k2*t) kernel"""

        s = 1 / k2
        f1 = PlateauFunction()
        f1.scale = s

        f2 = ExpDecayFunction(k2)
        f2.scale = -s

        return IntegrableSumFunction([f1, f2])


