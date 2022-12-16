import abc
import types
import numpy as np
import numpy.typing as npt
import pynufft

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
                 input_dtype: type | None = None,
                 output_dtype: type | None = None) -> None:
        """Linear operator abstract base class that maps real or complex array x to y

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

        if input_dtype is not None:
            self._input_dtype = input_dtype
        else:
            self._input_dtype = xp.float64

        if output_dtype is not None:
            self._output_dtype = output_dtype
        else:
            self._output_dtype = xp.float64

    @property
    def input_dtype(self) -> type:
        """the data type of the input array"""
        return self._input_dtype

    @property
    def output_dtype(self) -> type:
        """the data type of the output array"""
        return self._output_dtype

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
        x = self.xp.random.rand(*self.input_shape).astype(self.input_dtype)
        y = self.xp.random.rand(*self.output_shape).astype(self.output_dtype)

        if self.xp.iscomplexobj(x):
            x += 1j * self.xp.random.rand(*self.input_shape).astype(
                self.input_dtype)

        if self.xp.iscomplexobj(y):
            y += 1j * self.xp.random.rand(*self.output_shape).astype(
                self.output_dtype)

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

        x = self.xp.random.rand(*self._input_shape).astype(self.input_dtype)

        if self.xp.iscomplexobj(x):
            x += 1j * self.xp.random.rand(*self.input_shape).astype(
                self.input_dtype)

        for i in range(num_iter):
            x = self.adjoint(self.forward(x))
            n = self.xp.linalg.norm(x.ravel())
            x /= n

        return self.xp.sqrt(n)

    def unravel_pseudo_complex(
            self, x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """unravel a real flattened pseudo-complex array into a complex array

        Parameters
        ----------
        x : npt.NDArray | cpt.NDArray
            real flattened array with size 2*prod(input_shape)

        Returns
        -------
        npt.NDArray | cpt.NDArray
            unraveled complex array
        """

        x = x.reshape(self.input_shape + (2, ))

        if x.dtype == self.xp.float64:
            return self.xp.squeeze(x.view(dtype=self.xp.complex128), axis=-1)
        elif x.dtype == self.xp.float32:
            return self.xp.squeeze(x.view(dtype=self.xp.complex64), axis=-1)
        elif x.dtype == self.xp.float128:
            return self.xp.squeeze(x.view(dtype=self.xp.complex256), axis=-1)
        else:
            raise ValueError(
                'Input must have dtyoe float32, float64 or float128')

    def ravel_pseudo_complex(
            self, x: npt.NDArray | cpt.NDArray) -> npt.NDArray | cpt.NDArray:
        """ravel a complex array into a flattened pesudo complex array

        Parameters
        ----------
        x : npt.NDArray | cpt.NDArray
            complex array of shape input_shape
        Returns
        -------
        npt.NDArray | cpt.NDArray
            ravel pesudo complex array of shape 2*prod(input_shape)
        """
        return self.xp.stack([x.real, x.imag], axis=-1).ravel()


class FFT(LinearOperator):
    """ fast fourier transform operator matched to replicated continous FT"""

    def __init__(self,
                 x: npt.NDArray,
                 xp: types.ModuleType = np,
                 dtype: type | None = None) -> None:

        if dtype is None:
            dtype = xp.complex128

        super().__init__(input_shape=x.shape,
                         output_shape=x.shape,
                         xp=xp,
                         input_dtype=dtype,
                         output_dtype=dtype)

        self._dx = float(x[1] - x[0])
        self._x = x
        self._phase_factor = self.dx * xp.exp(-1j * self.k * float(x[0]))
        self._scale_factor = float(np.sqrt(x.size / (2 * np.pi)))
        self._adjoint_factor = float((np.abs(x[0])**2) / ((x.size / 2)**2))

    @property
    def x(self) -> npt.NDArray:
        return self._x

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def k(self) -> npt.NDArray:
        return self.xp.fft.fftfreq(self.x.size, d=self.dx) * 2 * self.xp.pi

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

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        return self.xp.fft.fft(
            x, norm='ortho') * self.phase_factor * self.scale_factor

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        return self.xp.fft.ifft(y * self.scale_factor / self.phase_factor,
                                norm='ortho') * self._adjoint_factor

    def inverse(self, y: npt.NDArray) -> npt.NDArray:
        return self.adjoint(y) / (self.scale_factor**2) / self.adjoint_factor


class T2CorrectedFFT(FFT):
    """ fast fourier transform operator matched to replicated continous FT"""

    def __init__(self,
                 x: npt.NDArray,
                 t_readout: npt.NDArray,
                 T2star: npt.NDArray,
                 xp: types.ModuleType = np,
                 dtype: type | None = None) -> None:
        super().__init__(x=x, xp=xp, dtype=dtype)

        self._t_readout = t_readout
        self._T2star = T2star

        # precalculate the decay envelopes at the readout times
        # assumes that readout time is a function of abs(k)
        self._n = self.x.shape[0]
        self._decay_envs = self.xp.zeros((self._n // 2 + 1, self._n))
        self._masks = self.xp.zeros((self._n // 2 + 1, self._n))
        inds = self.xp.where(T2star > 0)
        for i, t in enumerate(t_readout[:(self._n // 2 + 1)]):
            tmp = self.xp.zeros(self._n)
            tmp[inds] = (t / T2star[inds])
            self._decay_envs[i, :] = xp.exp(-tmp)
            self._masks[i, i] = 1
            self._masks[i, -i] = 1

    def forward(self, x: npt.NDArray) -> npt.NDArray:
        y = self.xp.zeros(self._n, dtype=self.output_dtype)

        for i in range(self._n // 2 + 1):
            y += super().forward(
                self._decay_envs[i, :] * x) * self._masks[i, :]

        return y

    def adjoint(self, y: npt.NDArray) -> npt.NDArray:
        x = self.xp.zeros(self._n, dtype=self.input_dtype)

        for i in range(self._n // 2 + 1):
            x += super().adjoint(
                self._masks[i, :] * y) * self._decay_envs[i, :]

        return x


class GradientOperator(LinearOperator):
    """finite difference gradient operator for real or complex arrays"""

    def __init__(self,
                 input_shape: tuple[int, ...],
                 xp: types.ModuleType = np,
                 dtype: type | None = None) -> None:
        """_summary_

        Parameters
        ----------
        input_shape : tuple[int, ...]
            the input array shape
        xp : types.ModuleType
            the array module (numpy or cupy)
        dtype : type, optional,
            data type of the input array, by default float64
        """

        if dtype is None:
            dtype = xp.float64

        output_shape = (len(input_shape), ) + input_shape
        super().__init__(input_shape,
                         output_shape,
                         xp=xp,
                         input_dtype=dtype,
                         output_dtype=dtype)

    def forward(self, x):
        g = self.xp.zeros(self.output_shape, dtype=self.output_dtype)
        for i in range(x.ndim):
            g[i, ...] = self.xp.diff(x,
                                     axis=i,
                                     append=self.xp.take(x, [-1], i))

        return g

    def adjoint(self, y):
        d = self.xp.zeros(self.input_shape, dtype=self.input_dtype)

        for i in range(y.shape[0]):
            tmp = y[i, ...]
            sl = [slice(None)] * y.shape[0]
            sl[i] = slice(-1, None)
            tmp[tuple(sl)] = 0
            d -= self.xp.diff(tmp, axis=i, prepend=0)

        return d


class MultiChannelNonCartesianMRAcquisitionModel(LinearOperator):
    """acquisition model for multi channel MR with non cartesian stacked sampling using pynufft"""

    def __init__(self,
                 input_shape: tuple[int, ...],
                 coil_sensitivities: npt.NDArray,
                 k_space_sample_points: npt.NDArray,
                 interpolation_size: None | tuple[int, ...] = None,
                 scaling_factor: float = 1.,
                 device_number=0) -> None:
        """
        Parameters 
        ----------
 
        input_shape : tuple[int, ...]
            shape of the complex input image
        coil_sensitivities : npt.NDArray
            the complex coil sensitivities, shape (num_channels, image_shape)
        interpolation_size: tuple(int,int,int), optional
            interpolation size for nufft, default None which means 6 in all directions
        scaling_factor: float, optional
            extra scaling factor applied to the adjoint, default 1
        device_number: int, optional
            device from pynuffts device list to use, default 0
        """

        self._coil_sensitivities = coil_sensitivities
        self._num_channels = coil_sensitivities.shape[0]
        self._scaling_factor = scaling_factor

        self._kspace_sample_points = k_space_sample_points

        # size of the oversampled kspace grid
        self._Kd = tuple(2 * x for x in input_shape)
        # the adjoint from pynufft needs to be scaled by this factor
        self._adjoint_scaling_factor = np.prod(self._Kd)

        if interpolation_size is None:
            self._interpolation_size = len(input_shape) * (6, )
        else:
            self._interpolation_size = interpolation_size

        self._device_number = device_number
        self._device = pynufft.helper.device_list()[self._device_number]

        # setup a nufft object for every stack
        self._nufft = pynufft.NUFFT(self._device)
        self._nufft.plan(self.kspace_sample_points, input_shape, self._Kd,
                         self._interpolation_size)

        super().__init__(input_shape=input_shape,
                         output_shape=(self._num_channels,
                                       self._kspace_sample_points.shape[0]),
                         xp=np,
                         input_dtype=np.complex64,
                         output_dtype=np.complex64)

    def __del__(self) -> None:
        del self._nufft

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
    def coil_sensitivities(self) -> npt.NDArray:
        """
        Returns
        -------
        npt.NDArray
            array of coil sensitivities
        """
        return self._coil_sensitivities

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

        y = np.zeros(self.output_shape, dtype=self.output_dtype)

        for i in range(self._num_channels):
            y[i,
              ...] = self._nufft.forward(self.coil_sensitivities[i, ...] * x)

        if self._scaling_factor != 1:
            y *= self._scaling_factor

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
        x = np.zeros(self.input_shape, dtype=self.input_dtype)

        for i in range(self._num_channels):
            x += np.conj(
                self.coil_sensitivities[i, ...]) * self._nufft.adjoint(
                    y[i, ...]) * self._adjoint_scaling_factor

        if self._scaling_factor != 1:
            x *= self._scaling_factor

        return x


class MultiChannelStackedNonCartesianMRAcquisitionModel(LinearOperator):
    """acquisition model for multi channel MR with non cartesian stacked sampling using pynufft"""

    def __init__(self,
                 input_shape: tuple[int, int, int],
                 coil_sensitivities: npt.NDArray,
                 k_space_sample_points: npt.NDArray,
                 interpolation_size: None | tuple[int, int] = None,
                 scaling_factor: float = 1.,
                 device_number=0) -> None:
        """
        Parameters 
        ----------
 
        input_shape : tuple[int, ...]
            shape of the complex input image
        coil_sensitivities : npt.NDArray
            the complex coil sensitivities, shape (num_channels, image_shape)
        k_space_sample_points, npt.NDArray
            2D coordinates of kspace sample points
        interpolation_size: tuple(int,int), optional
            interpolation size for nufft, default None which means 6 in all directions
        scaling_factor: float, optional
            extra scaling factor applied to the adjoint, default 1
        device_number: int, optional
            device from pynuffts device list to use, default 0
        """

        self._coil_sensitivities = coil_sensitivities
        self._num_channels = coil_sensitivities.shape[0]
        self._scaling_factor = scaling_factor

        self._kspace_sample_points = k_space_sample_points

        # size of the oversampled kspace grid
        self._Kd = tuple(2 * x for x in input_shape[1:])
        # the adjoint from pynufft needs to be scaled by this factor
        self._adjoint_scaling_factor = np.prod(self._Kd)

        if interpolation_size is None:
            self._interpolation_size = (6, 6)
        else:
            self._interpolation_size = interpolation_size

        self._device_number = device_number
        self._device = pynufft.helper.device_list()[self._device_number]

        # setup a nufft object for every stack
        self._nufft_2d = pynufft.NUFFT(self._device)
        self._nufft_2d.plan(self.kspace_sample_points, input_shape[1:],
                            self._Kd, self._interpolation_size)

        super().__init__(input_shape=input_shape,
                         output_shape=(self._num_channels, input_shape[0],
                                       self._kspace_sample_points.shape[0]),
                         xp=np,
                         input_dtype=np.complex64,
                         output_dtype=np.complex64)

    def __del__(self) -> None:
        del self._nufft_2d

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
    def coil_sensitivities(self) -> npt.NDArray:
        """
        Returns
        -------
        npt.NDArray
            array of coil sensitivities
        """
        return self._coil_sensitivities

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

        y = np.zeros(self.output_shape, dtype=self.output_dtype)

        for i in range(self._num_channels):
            # perform a 1D FFT along the "stack axis"
            tmp = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(
                self.coil_sensitivities[i, ...] * x, axes=0),
                                              axes=[0]),
                                  axes=0)

            # series of 2D NUFFTs
            for k in range(self.input_shape[0]):
                y[i, k, ...] = self._nufft_2d.forward(tmp[k, ...])

        if self._scaling_factor != 1:
            y *= self._scaling_factor

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
        x = np.zeros(self.input_shape, dtype=self.input_dtype)

        for i in range(self._num_channels):
            tmp = self.xp.zeros(self.input_shape, dtype=self.input_dtype)
            # series of 2D adjoint NUFFTs
            for k in range(self.input_shape[0]):
                tmp[k, ...] = self._nufft_2d.adjoint(y[i, k, ...])

            x += np.conj(self.coil_sensitivities[i, ...]) * np.fft.ifftshift(
                np.fft.ifftn(np.fft.ifftshift(tmp, axes=0), axes=[0]), axes=0)

        # when using numpy's fftn with the default normalization
        # we have to multiply the inverse with input_shape[0] to get the adjoint
        x *= (self._adjoint_scaling_factor * self.input_shape[0])

        if self._scaling_factor != 1:
            x *= self._scaling_factor

        return x
