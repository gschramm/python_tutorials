import abc
import numpy as np
import numpy.typing as npt
import pynufft
import matplotlib.pyplot as plt


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

    def __init__(self,
                 x_shape: tuple,
                 y_shape: tuple,
                 flat_mode: bool = False) -> None:
        """Linear operator abstract base class that maps real array x to real array y

        Parameters
        ----------
        x_shape : tuple
            shape of x array
        y_shape : tuple
            shape of y array
        flat_mode : bool, optional
            whether input array x is passed as flattened array
        """
        super().__init__()

        self._x_shape = x_shape
        self._y_shape = y_shape
        self._flat_mode = flat_mode

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

    @property
    def flat_mode(self) -> bool:
        return self._flat_mode

    @flat_mode.setter
    def flat_mode(self, value: bool) -> None:
        self._flat_mode = value

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
        if self.flat_mode:
            x = np.random.rand(np.prod(self._x_shape))
            y = np.random.rand(np.prod(self._y_shape))
        else:
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


class IdentityOperator(LinearOperator):
    """Identity operator"""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Identity operator

        Parameters
        ----------
        shape : tuple[int, ...]
            shape of the array where Identity operates on
        """
        super().__init__(shape, shape)

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        return x

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        return y


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
        if self.flat_mode:
            x = x.reshape(self.x_shape)

        x_complex = complex_view_of_real_array(x)
        y = np.zeros(self._y_shape, dtype=x.dtype)

        for i in range(self._num_channels):
            y[i, ...] = real_view_of_complex_array(
                np.fft.fftn(self._coil_sensitivites_complex[i, ...] *
                            x_complex,
                            norm='ortho'))

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
                np.conj(self._coil_sensitivites_complex[i, ...]) *
                np.fft.ifftn(y_complex[i, ...], norm='ortho'))

        if self.flat_mode:
            x = x.ravel()
        return x


class MultiChannelNonCartesianMRAcquisitionModel(LinearOperator):
    """acquisition model for multi channel MR with non cartesian stacked sampling using pynufft"""

    def __init__(self,
                 image_shape: tuple[int, ...],
                 coil_sensitivities: npt.NDArray,
                 k_space_sample_points: npt.NDArray,
                 interpolation_size: None | tuple[int, ...] = None,
                 scaling_factor: float = 1.,
                 device_number=0) -> None:
        """
        Parameters 
 
        image_shape : tuple[int, ...]
            shape of the input image excluding the pseudo-complex axis
        coil_sensitivities : npt.NDArray
            the (pseudo-complex) coil sensitivities, shape (num_channels, image_shape, 2)
        interpolation_size: tuple(int,int,int), optional
            interpolation size for nufft, default None which means 6 in all directions
        scaling_factor: float, optional
            extra scaling factor applied to the adjoint, default 1
        device_number: int, optional
            device from pynuffts device list to use, default 0
        """

        x_shape = image_shape + (2, )

        self._coil_sensitivites_complex = complex_view_of_real_array(
            coil_sensitivities)
        self._num_channels = coil_sensitivities.shape[0]
        self._scaling_factor = scaling_factor

        self._kspace_sample_points = k_space_sample_points

        # size of the oversampled kspace grid
        self._Kd = tuple(2 * x for x in image_shape)
        # the adjoint from pynufft needs to be scaled by this factor
        self._adjoint_scaling_factor = np.prod(self._Kd)

        if interpolation_size is None:
            self._interpolation_size = len(image_shape) * (6, )
        else:
            self._interpolation_size = interpolation_size

        self._device_number = device_number
        self._device = pynufft.helper.device_list()[self._device_number]

        # setup a nufft object for every stack
        self._nufft = pynufft.NUFFT(self._device)
        self._nufft.plan(self.kspace_sample_points, image_shape, self._Kd,
                         self._interpolation_size)

        super().__init__(
            x_shape,
            (self._num_channels, self._kspace_sample_points.shape[0], 2))

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

    @property
    def kspace_sample_points(self) -> npt.NDArray:
        """
        Returns
        -------
        npt.NDArray
            the kspace sample points
        """
        return self._kspace_sample_points

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        """forward method

        Parameters
        ----------
        x : npt.NDArray
            (pseudo-complex) image with shape (image_shape,2)

        Returns
        -------
        npt.NDArray
            (pseudo-complex) data y with shape (num_channels,num_kspace_points,2)
        """

        # if x is passed as flattened / ravel array, we have to reshape it
        # this can e.g. happen when using scipy's fmin_cg
        if self.flat_mode:
            x = x.reshape(self.x_shape)

        x_complex = complex_view_of_real_array(x)
        y = np.zeros(self._y_shape, dtype=x.dtype)

        for i in range(self._num_channels):
            y[i, ...] = real_view_of_complex_array(
                self._nufft.forward(self._coil_sensitivites_complex[i, ...] *
                                    x_complex))

        if self._scaling_factor != 1:
            y *= self._scaling_factor

        if self.flat_mode:
            y = y.ravel()
        return y

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        """adjoint of forward method

        Parameters
        ----------
        y : npt.NDArray
            (pseudo-complex) data with shape (num_channels,num_kspace_points,2)

        Returns
        -------
        npt.NDArray
            (pseudo-complex) image x with shape (image_shape,2)
        """
        if self.flat_mode:
            y = y.reshape(self.y_shape)
        y_complex = complex_view_of_real_array(y)
        x = np.zeros(self._x_shape, dtype=y.dtype)

        for i in range(self._num_channels):
            x += real_view_of_complex_array(
                np.conj(self._coil_sensitivites_complex[i, ...]) *
                self._nufft.adjoint(y_complex[i, ...]) *
                self._adjoint_scaling_factor)

        if self._scaling_factor != 1:
            x *= self._scaling_factor

        if self.flat_mode:
            x = x.ravel()
        return x

    def show_kspace_sample_points(self, **kwargs) -> plt.figure:
        """show kspace sample points in a 3D scatter plot

        Parameters
        ----------

        kwargs: dict
            keyword arguments passed to plt.scatter()

        Returns
        -------
        plt.figure
            figure containing the scatter plot
        """

        if self._kspace_sample_points.shape[1] == 2:
            fig, ax = plt.subplots()
            ax.plot(self.kspace_sample_points[:, 0],
                    self._kspace_sample_points[:, 1])
            fig.tight_layout()
            fig.show()

        elif self._kspace_sample_points.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(self.kspace_sample_points[:, 0],
                       self.kspace_sample_points[:, 1],
                       self.kspace_sample_points[:, 2],
                       marker='.',
                       **kwargs)
            fig.tight_layout()
            fig.show()

        return fig


class ComplexGradientOperator(LinearOperator):
    """finite forward difference gradient operator for pseudo complex arrays (2 real arrays)"""

    def __init__(self, n: int | tuple[int, int, int], ndim: int) -> None:
        """
        Parameters
        ----------
        n : int | tuple[int,int,int]
            spatial shape
            if an integer is given, the shape is (n,n,n)
        ndim : int
            number of dimensions
            the image x has the dimension (n0,n1,n2) + (2,)
            the gradient image y has the dimension (ndim,) + (n0,n1,n2) + (2,)
        """
        self._ndim = ndim
        self._sl_first = slice(0, 1, None)
        self._sl_last = slice(-1, None, None)
        self._sl_all = slice(None, None, None)

        if isinstance(n, int):
            x_shape = self._ndim * (n, )
        else:
            x_shape = n

        super().__init__(x_shape + (2, ), (self._ndim, ) + x_shape + (2, ))

    @property
    def ndim(self) -> int:
        return self._ndim

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        # if x is passed as flattened / ravel array, we have to reshape it
        # this can e.g. happen when using scipy's fmin_cg
        if self.flat_mode:
            x = x.reshape(self.x_shape)

        g = np.zeros(self.y_shape, dtype=x.dtype)

        # we use numpy's diff functions and append/prepend the first/last slice
        for i in range(self._ndim):
            sl = i * (self._sl_all, ) + (
                self._sl_last, ) + (self._ndim - i - 1) * (self._sl_all, )
            g[i, ..., 0] = np.diff(x[..., 0], axis=i, append=x[..., 0][sl])
            g[i, ..., 1] = np.diff(x[..., 1], axis=i, append=x[..., 1][sl])

        if self.flat_mode:
            g = g.ravel()

        return g

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        if self.flat_mode:
            y = y.reshape(self.y_shape)

        d = np.zeros(self.x_shape, dtype=y.dtype)

        for i in range(self._ndim):
            sl = i * (self._sl_all, ) + (
                self._sl_last, ) + (self._ndim - i - 1) * (self._sl_all, )
            tmp0 = y[i, ..., 0]
            tmp0[sl] = 0
            d[..., 0] -= np.diff(tmp0, axis=i, prepend=0)
            tmp1 = y[i, ..., 1]
            tmp1[sl] = 0
            d[..., 1] -= np.diff(tmp1, axis=i, prepend=0)

        if self.flat_mode:
            d = d.ravel()
        return d