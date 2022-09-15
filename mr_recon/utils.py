import abc
import numpy as np
import numpy.typing as npt


def complex_view_of_real_array(x: npt.NDArray) -> npt.NDArray:
    """return a complex view of of a real array, by interpreting the last dimension as as real and imaginary part
       output = input[...,0] + 1j * input[....,1]
    """
    if x.dtype == np.float64:
        return np.squeeze(x.view(dtype=np.complex128), axis=-1)
    elif x.dtype == np.float32:
        return np.squeeze(x.view(dtype=np.complex64), axis=-1)
    elif x.dtype == np.float128:
        return np.squeeze(x.view(dtype=np.complex256), axis=-1)
    else:
        raise ValueError('Input must have dtyoe float32, float64 or float128')


def real_view_of_complex_array(x: npt.NDArray) -> npt.NDArray:
    """return a real view of a complex array
       output[...,0] = real(input)
       output[...,1] = imaginary(input)
    """
    return np.stack([x.real, x.imag], axis=-1)


class LinearOperator(abc.ABC):

    def __init__(self, x_shape: tuple, y_shape: tuple) -> None:
        """Linear operator abstract base class that maps real array x to real array y

        Parameters
        ----------
        x_shape : tuple
            shape of x array
        y_shape : tuple
            shape of y array
        """
        super().__init__()

        self._x_shape = x_shape
        self._y_shape = y_shape

    @property
    def x_shape(self) -> tuple:
        """shape of x array

        Returns
        -------
        tuple
            shape of x array
        """
        return self._x_shape

    @property
    def y_shape(self):
        """shape of y array

        Returns
        -------
        tuple
            shape of y array
        """
        return self._y_shape

    @abc.abstractmethod
    def forward(self, x: npt.NDArray) -> npt.NDArray:
        """forward step

        Parameters
        ----------
        x : npt.NDArray
            x array

        Returns
        -------
        npt.NDArray
            the linear operator applied to x
        """
        pass

    @abc.abstractmethod
    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        """adjoint of forward step

        Parameters
        ----------
        y : npt.NDArray
            y array

        Returns
        -------
        npt.NDArray
            the adjoint of the linear operator applied to y
        """
        raise NotImplementedError()

    def adjointness_test(self) -> None:
        """test if adjoint is really the adjoint of forward
        """
        x = np.random.rand(*self._x_shape)
        y = np.random.rand(*self._y_shape)

        x_fwd = self.forward(x)
        y_back = self.adjoint(y)

        assert (np.isclose((x_fwd * y).sum(), (x * y_back).sum()))

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

        x = np.random.rand(*self._x_shape)

        for i in range(num_iter):
            x = self.adjoint(self.forward(x))
            n = np.linalg.norm(x.ravel())
            x /= n

        return np.sqrt(n)


class MatrixLinearOperator(LinearOperator):

    def __init__(self, A: npt.NDArray) -> None:
        super().__init__((A.shape[1], ), (A.shape[0], ))
        self._A = A

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        return self._A @ x

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        return self._A.T @ y


class MultiChannel3DCartesianMRAcquisitionModel(LinearOperator):
    """acquisition model for multi channel MR with cartesian sampling"""

    def __init__(self, n: int, num_channels: int,
                 coil_sensitivities: npt.NDArray) -> None:
        """
        Parameters
        ----------
        n : int
            spatial dimension
            the (pseudo-complex) image x has shape (n,n,n,2)
            the (pseudo-complex) data y has shape (num_channels,n,n,n,2)
        num_channels : int
            number of channels (coils)
        coil_sensitivities : npt.NDArray
            the (pseudo-complex) coil sensitivities, shape (num_channels,n,n,n,2)
        """
        super().__init__((n, n, n, 2), (num_channels, n, n, n, 2))

        self._n = n
        self._num_channels = num_channels
        self._coil_sensitivites_complex = complex_view_of_real_array(
            coil_sensitivities)

    @property
    def n(self) -> int:
        """
        Returns
        -------
        int
            spatial dimension
        """
        return self._n

    @property
    def num_channels(self) -> int:
        """
        Returns
        -------
        int
            number of channels (coils)
        """
        return self._num_channels

    @property
    def coil_sensivities(self) -> npt.NDArray:
        """
        Returns
        -------
        npt.NDArray
            (pseud-complex) array of coil sensitivities
        """
        return real_view_of_complex_array(self._coil_sensitivites_complex)

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        """forward method

        Parameters
        ----------
        x : npt.NDArray
            (pseudo-complex) image with shape (n,n,n,2)

        Returns
        -------
        npt.NDArray
            (pseudo-complex) data y with shape (num_channels,n,n,n,2)
        """
        x_complex = complex_view_of_real_array(x)
        y = np.zeros(self._y_shape, dtype=x.dtype)

        for i in range(self._num_channels):
            y[i, ...] = real_view_of_complex_array(
                self._coil_sensitivites_complex[i, ...] *
                np.fft.fftn(x_complex, norm='ortho'))

        return y

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        """adjoint of forward method

        Parameters
        ----------
        y : npt.NDArray
            (pseudo-complex) data with shape (num_channels,n,n,n,2)

        Returns
        -------
        npt.NDArray
            (pseudo-complex) image x with shape (n,n,n,2)
        """
        y_complex = complex_view_of_real_array(y)
        x = np.zeros(self._x_shape, dtype=y.dtype)

        for i in range(self._num_channels):
            x += real_view_of_complex_array(
                np.fft.ifftn(np.conj(self._coil_sensitivites_complex[i, ...]) *
                             y_complex[i, ...],
                             norm='ortho'))

        return x


class ComplexGradientOperator(LinearOperator):
    """finite forward difference gradient operator for pseudo complex arrays (2 real arrays)"""

    def __init__(self, n: int, ndim: int) -> None:
        """
        Parameters
        ----------
        n : int
            spatial dimension
        ndim : int
            number of dimensions
            the image x has the dimension ndim(n,) + (2,)
            the gradient image y has the dimension (ndim,) + ndim(n,) + (2,)
        """
        self._ndim = ndim
        self._sl_first = slice(0, 1, None)
        self._sl_last = slice(-1, None, None)
        self._sl_all = slice(None, None, None)

        super().__init__(self._ndim * (n, ) + (2, ),
                         (self._ndim, ) + self._ndim * (n, ) + (2, ))

    @property
    def ndim(self) -> int:
        return self._ndim

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        g = np.zeros(self._y_shape, dtype=x.dtype)

        # we use numpy's diff functions and append/prepend the first/last slice
        for i in range(self._ndim):
            sl = i * (self._sl_all, ) + (
                self._sl_last, ) + (self._ndim - i - 1) * (self._sl_all, )
            g[i, ..., 0] = np.diff(x[..., 0], axis=i, append=x[..., 0][sl])
            g[i, ..., 1] = np.diff(x[..., 1], axis=i, append=x[..., 1][sl])

        return g

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        d = np.zeros(self._x_shape, dtype=y.dtype)

        for i in range(self._ndim):
            sl = i * (self._sl_all, ) + (
                self._sl_first, ) + (self._ndim - i - 1) * (self._sl_all, )
            d[..., 0] -= np.diff(y[i, ..., 0],
                                 axis=i,
                                 prepend=y[i, ..., 0][sl])
            d[..., 1] -= np.diff(y[i, ..., 1],
                                 axis=i,
                                 prepend=y[i, ..., 1][sl])

        return d


class Norm(abc.ABC):
    """abstract base clase for norms where we can calculate the prox of the convex dual"""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abc.abstractmethod
    def __call__(self, x: npt.NDArray) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def prox_convex_dual(self, x: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError


class ComplexGradientNorm(Norm):
    """norm of a pseudo-complex gradient field - real and imaginary part are treated separately"""

    def __init__(self, name: str = 'l2_l1') -> None:
        """
        Parameters
        ----------

        name : str
          name of the norm
          'l2_l1' ... mixed L2/L1 (sum of pointwise Euclidean norms in every voxel)
          'l2_sq' ... squared l2 norm (sum of pointwise squared Euclidean norms in every voxel)
        """
        super().__init__(name)

        if not self._name in ['l2_l1', 'l2_sq']:
            raise NotImplementedError

    def __call__(self, x: npt.NDArray) -> float:
        """
        Parameters
        ----------
        x : npt.NDArray
            complex gradient of pseudo-complex array

        Returns
        -------
        float
            the complex gradient norm
        """
        if self._name == 'l2_l1':
            n = np.linalg.norm(x[..., 0], axis=0).sum() + np.linalg.norm(
                x[..., 1], axis=0).sum()
        elif self._name == 'l2_sq':
            n = (x[..., 0]**2).sum() + (x[..., 1]**2).sum()

        return n

    def prox_convex_dual(self,
                         x: npt.NDArray,
                         sigma: float = 1.) -> npt.NDArray:
        """proximal operator of the convex dual of the norm

        Parameters
        ----------
        x : npt.NDArray
            complex gradient of pseudo-complex array
        sigma : float, optional
            sigma parameter of prox, by default 1

        Returns
        -------
        npt.NDArray
            the proximity operator of the convex dual of the norm applied on x
        """
        if self._name == 'l2_l1':
            gnorm0 = np.linalg.norm(x[..., 0], axis=0)
            r0 = x[..., 0] / np.clip(gnorm0, 1, None)

            gnorm1 = np.linalg.norm(x[..., 1], axis=0)
            r1 = x[..., 1] / np.clip(gnorm1, 1, None)

        elif self._name == 'l2_sq':
            r0 = x[..., 0] / (1 + sigma)
            r1 = x[..., 1] / (1 + sigma)

        return np.stack([r0, r1], axis=-1)


if __name__ == '__main__':
    np.random.seed(1)
    n = 64
    num_channels = 4
    coil_sens = np.random.rand(num_channels, n, n, n, 2)
    mop = MultiChannel3DCartesianMRAcquisitionModel(n, num_channels, coil_sens)
    mop.adjointness_test()

    gop = ComplexGradientOperator(n, 3)