import numpy as np
import numpy.typing as npt


class FFT:
    """ fast fourier transform operator matched to replicated continous FT"""

    def __init__(self, x: npt.NDArray) -> None:
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
