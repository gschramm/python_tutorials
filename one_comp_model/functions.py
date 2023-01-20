import abc
import functools
import numpy as np
import numpy.typing as npt

from copy import deepcopy

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



#-------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    K1_high = 0.6
    K1_low = 0.2

    Vt = 1.0
    fbv = 0.05

    k2_high = K1_high/Vt
    k2_low = K1_low/Vt


    g11 = ExpDecayFunction(4)
    g11.scale = 50.0
    g12 = ExpDecayFunction(8)
    g12.scale = -53.0
    g13 = ExpDecayFunction(0.5)
    g13.scale = 2.0
    p1 = PlateauFunction()
    p1.scale = 1.0

    # generate an arterial input function as sum of 3 exponentials + a plateau
    C_A1 = ExpConvSumFunction([g11,g12,g13,p1])

    # generate a second input function where the "peak/plateau" ratio is lower
    g21 = ExpDecayFunction(4)
    g21.scale = 37.0
    g22 = ExpDecayFunction(8)
    g22.scale = -49.0
    g23 = ExpDecayFunction(0.5)
    g23.scale = 2.0
    p2 = deepcopy(p1)
    p2.scale = 10.0

    C_A2 = ExpConvSumFunction([g21,g22,g23,p2])

    # constant infuction IF 
    C_A3 = ExpConvSumFunction([p2])
    
    # tissue response = K1 * convolution(C_A, exp(-k2*t))
    C_t1_high = C_A1.expconv(k2_high)
    C_t1_high.scale = K1_high
    C_t1_low = C_A1.expconv(k2_low)
    C_t1_low.scale = K1_low

    C_t2_high = C_A2.expconv(k2_high)
    C_t2_high.scale = K1_high
    C_t2_low = C_A2.expconv(k2_low)
    C_t2_low.scale = K1_low

    C_t3_high = C_A3.expconv(k2_high)
    C_t3_high.scale = K1_high
    C_t3_low = C_A3.expconv(k2_low)
    C_t3_low.scale = K1_low

    # calculate PET concentrations including fractional blood volume
    scaled_CA1 = deepcopy(C_A1)
    scaled_CA1.scale *= 0.05
    tmp1_high = deepcopy(C_t1_high)
    tmp1_high.scale *= 1-fbv
    tmp1_low = deepcopy(C_t1_low)
    tmp1_low.scale *= 1-fbv

    C_PET1_high = IntegrableSumFunction([scaled_CA1, tmp1_high])
    C_PET1_low = IntegrableSumFunction([scaled_CA1, tmp1_low])

    scaled_CA2 = deepcopy(C_A2)
    scaled_CA2.scale *= 0.05
    tmp2_high = deepcopy(C_t2_high)
    tmp2_high.scale *= 1-fbv
    tmp2_low = deepcopy(C_t2_low)
    tmp2_low.scale *= 1-fbv

    C_PET2_high = IntegrableSumFunction([scaled_CA2, tmp2_high])
    C_PET2_low = IntegrableSumFunction([scaled_CA2, tmp2_low])

    scaled_CA3 = deepcopy(C_A3)
    scaled_CA3.scale *= 0.05
    tmp3_high = deepcopy(C_t3_high)
    tmp3_high.scale *= 1-fbv
    tmp3_low = deepcopy(C_t3_low)
    tmp3_low.scale *= 1-fbv

    C_PET3_high = IntegrableSumFunction([scaled_CA3, tmp3_high])
    C_PET3_low = IntegrableSumFunction([scaled_CA3, tmp3_low])

    # discrete time array for plots
    t = np.linspace(0.001, 8, 1000)

    fig, ax = plt.subplots(3, 4, figsize=(12,9), sharex = True)

    ax[0,0].plot(t, C_A1(t), 'k', label = 'C_A')
    ax[0,0].plot(t, C_t1_high(t), 'r', label = f'C_t(K1={K1_high})')
    ax[0,0].plot(t, C_t1_low(t), 'b', label = f'C_t(K1={K1_low})')
    ax[0,0].plot(t, C_PET1_high(t), 'r:', label = f'C_PET(K1={K1_high})')
    ax[0,0].plot(t, C_PET1_low(t), 'b:', label = f'C_PET(K1={K1_low})')

    ax[0,1].plot(t, C_t1_high(t)/C_t1_low(t), 'k')
    ax[0,1].plot(t, C_PET1_high(t)/C_PET1_low(t), 'k:')

    ax[0,2].plot(t, C_t1_high.indefinite_integral(t) - C_t1_high.indefinite_integral(0), 'r')
    ax[0,2].plot(t, C_t1_low.indefinite_integral(t) - C_t1_low.indefinite_integral(0), 'b')
    ax[0,2].plot(t, C_PET1_high.indefinite_integral(t) - C_PET1_high.indefinite_integral(0), 'r:')
    ax[0,2].plot(t, C_PET1_low.indefinite_integral(t) - C_PET1_low.indefinite_integral(0), 'b:')

    ax[0,3].plot(t, (C_t1_high.indefinite_integral(t) - C_t1_high.indefinite_integral(0)) / 
                    (C_t1_low.indefinite_integral(t) - C_t1_low.indefinite_integral(0)), 'k')
    ax[0,3].plot(t, (C_PET1_high.indefinite_integral(t) - C_PET1_high.indefinite_integral(0)) / 
                    (C_PET1_low.indefinite_integral(t) - C_PET1_low.indefinite_integral(0)), 'k:')

    ax[1,0].plot(t, C_A2(t), 'k', label = 'C_A')
    ax[1,0].plot(t, C_t2_high(t), 'r', label = f'C_t(K1={K1_high})')
    ax[1,0].plot(t, C_t2_low(t), 'b', label = f'C_t(K1={K1_low})')
    ax[1,0].plot(t, C_PET2_high(t), 'r:', label = f'C_PET(K1={K1_high})')
    ax[1,0].plot(t, C_PET2_low(t), 'b:', label = f'C_PET(K1={K1_low})')

    ax[1,1].plot(t, C_t2_high(t)/C_t2_low(t), 'k')
    ax[1,1].plot(t, C_PET2_high(t)/C_PET2_low(t), 'k:')

    ax[1,2].plot(t, C_t2_high.indefinite_integral(t) - C_t2_high.indefinite_integral(0), 'r')
    ax[1,2].plot(t, C_t2_low.indefinite_integral(t) - C_t2_low.indefinite_integral(0), 'b')

    ax[1,3].plot(t, (C_t2_high.indefinite_integral(t) - C_t2_high.indefinite_integral(0)) / 
                    (C_t2_low.indefinite_integral(t) - C_t2_low.indefinite_integral(0)), 'k')
    ax[1,3].plot(t, (C_PET2_high.indefinite_integral(t) - C_PET2_high.indefinite_integral(0)) / 
                    (C_PET2_low.indefinite_integral(t) - C_PET2_low.indefinite_integral(0)), 'k:')

    ax[2,0].plot(t, C_A3(t), 'k', label = 'C_A')
    ax[2,0].plot(t, C_t3_high(t), 'r', label = f'C_t(K1={K1_high})')
    ax[2,0].plot(t, C_t3_low(t), 'b', label = f'C_t(K1={K1_low})')
    ax[2,0].plot(t, C_PET3_high(t), 'r:', label = f'C_PET(K1={K1_high})')
    ax[2,0].plot(t, C_PET3_low(t), 'b:', label = f'C_PET(K1={K1_low})')

    ax[2,1].plot(t, C_t3_high(t)/C_t3_low(t), 'k')
    ax[2,1].plot(t, C_PET3_high(t)/C_PET3_low(t), 'k:')

    ax[2,2].plot(t, C_t3_high.indefinite_integral(t) - C_t3_high.indefinite_integral(0), 'r')
    ax[2,2].plot(t, C_t3_low.indefinite_integral(t) - C_t3_low.indefinite_integral(0), 'b')
    ax[2,3].plot(t, (C_t3_high.indefinite_integral(t) - C_t3_high.indefinite_integral(0)) / 
                    (C_t3_low.indefinite_integral(t) - C_t3_low.indefinite_integral(0)), 'k')
    ax[2,3].plot(t, (C_PET3_high.indefinite_integral(t) - C_PET3_high.indefinite_integral(0)) / 
                    (C_PET3_low.indefinite_integral(t) - C_PET3_low.indefinite_integral(0)), 'k:')

    ax[0,0].legend()
    for axx in ax.ravel():
        axx.grid(ls = ':')    
        axx.set_xlabel('t (min)')

    for axx in ax[:,1::2].ravel():
        axx.set_ylim(0, 1.05*K1_high/K1_low)

    fig.tight_layout()
    fig.show()