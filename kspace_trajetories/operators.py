import abc
import types
import numpy as np
import numpy.typing as npt

try:
    import cupy as cp
    import cupy.typing as cpt
except:
    import numpy as cp
    import numpy.typing as cpt


class LinearOperator(abc.ABC):

    def __init__(self,
                 input_shape: tuple[int, ...],
                 output_shape: tuple[int, ...],
                 xp: types.ModuleType,
                 dtype: type = float) -> None:
        """Linear operator abstract base class that maps real array x to real array y

        Parameters
        ----------
        input_shape : tuple
            shape of x array
        output_shape : tuple
            shape of y array
        xp : types.ModuleType
            module indicating whether to store all LOR endpoints as numpy as cupy array
        """

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._xp = xp
        self._dtype = dtype

    @property
    def dtype(self) -> type:
        """the data type of the input and output arrays"""
        return self._dtype

    @property
    def input_shape(self) -> tuple[int, ...]:
        """shape of x array

        Returns
        -------
        tuple
            shape of x array
        """
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        """shape of y array

        Returns
        -------
        tuple
            shape of y array
        """
        return self._output_shape

    @property
    def xp(self) -> types.ModuleType:
        """module indicating whether the LOR endpoints are stored as numpy or cupy array"""
        return self._xp

    @abc.abstractmethod
    def forward(self,
                x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """forward step

        Parameters
        ----------
        x : npt.NDArray | cpt.NDArray
            x array

        Returns
        -------
        npt.NDArray | cpt.NDArray
            the linear operator applied to x
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def adjoint(self,
                y: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """adjoint of forward step

        Parameters
        ----------
        y : npt.NDArray | cpt.NDArray
            y array

        Returns
        -------
        npt.NDArray | cpt.NDArray
            the adjoint of the linear operator applied to y
        """
        raise NotImplementedError()

    def adjointness_test(self, verbose=False) -> None:
        """test if adjoint is really the adjoint of forward

        Parameters
        ----------
        verbose : bool, optional
            prnt verbose output
        """
        x = self.xp.random.rand(*self.input_shape).astype(self.dtype)
        y = self.xp.random.rand(*self.output_shape).astype(self.dtype)

        if self.xp.iscomplexobj(x):
            x += 1j * self.xp.random.rand(*self.input_shape).astype(self.dtype)

        if self.xp.iscomplexobj(y):
            y += 1j * self.xp.random.rand(*self.output_shape).astype(
                self.dtype)

        x_fwd = self.forward(x)
        y_back = self.adjoint(y)

        a = (np.conj(y) * x_fwd).sum()
        b = (np.conj(y_back) * x).sum()

        if verbose:
            print(f'<y, A x>   {a}')
            print(f'<A^T y, x> {b}')

        assert (self.xp.isclose(a, b))

    def norm(self, num_iter=20) -> float:
        """estimate norm of operator via power iterations

        Parameters
        ----------
        num_iter : int, optional
            number of iterations, by default 20

        Returns
        -------
        float
            the estimated norm
        """

        x = self.xp.random.rand(*self._input_shape).astype(self.dtype)

        if self.xp.iscomplexobj(x):
            x += 1j * self.xp.random.rand(*self.input_shape).astype(self.dtype)

        for i in range(num_iter):
            x = self.adjoint(self.forward(x))
            n = self.xp.linalg.norm(x.ravel())
            x /= n

        return self.xp.sqrt(n)


class FFT(LinearOperator):
    """ fast fourier transform operator matched to replicated continous FT"""

    def __init__(self,
                 x: npt.NDArray,
                 xp: types.ModuleType = np,
                 dtype=complex) -> None:
        super().__init__(input_shape=x.shape,
                         output_shape=x.shape,
                         xp=xp,
                         dtype=dtype)
        self._dx = x[1] - x[0]
        self._x = x
        self._phase_factor = self.dx * np.exp(-1j * self.k * x[0])
        self._scale_factor = np.sqrt(x.size / (2 * np.pi))
        self._adjoint_factor = (np.abs(x[0])**2) / ((x.size / 2)**2)

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
