import numpy as np
import numpy.typing as npt

from scipy.ndimage import gaussian_filter


def rod_phantom(n: int = 256,
                r: float = 0.9,
                r_rod: float = 0.08,
                nrods: int = 5,
                rod_contrast: None | npt.NDArray = None) -> npt.NDArray:
    """3D cylinder rod phantom

    Parameters
    ----------
    n : int, optional
        spatial dimension, by default 256
    r : float, optional
        relative radius of cylinder, by default 0.9
    r_rod : float, optional
        reltative radius of rods , by default 0.08
    nrods : int, optional
        number of rods, by default 5
    rod_contrast : None | npt.NDArray, optional
        rod contrast, by default None

    Returns
    -------
    npt.NDArray
        the image array of shape (n,n,n)
    """

    if rod_contrast is None:
        rod_contrast = np.linspace(0.1, 1.2, nrods)

    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x, indexing='ij')

    R = np.sqrt((X - 0)**2 + (Y - 0)**2)

    x = np.zeros((n, n))

    x[R < r] = 1

    for i, ox in enumerate(
            np.linspace(-0.8 * r / np.sqrt(2), 0.8 * r / np.sqrt(2), nrods)):
        for j, oy in enumerate(
                np.linspace(-0.8 * r / np.sqrt(2), 0.8 * r / np.sqrt(2),
                            nrods)):
            R = np.sqrt((X - ox)**2 + (Y - oy)**2)
            x[R < r_rod] = rod_contrast[i]

    # repeat 2D arrays into 3D array
    x = np.repeat(x[:, :, np.newaxis], n, axis=2)

    # delete first / last slices
    x[:, :, :(n // 8)] = 0
    x[:, :, ((7 * n) // 8):] = 0

    return x


def generate_sensitivities(n: int, num_channels: int):
    x = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, x, indexing='ij')

    phis = np.linspace(0, 2 * np.pi, num=num_channels, endpoint=False)

    s = np.zeros((num_channels, n, n))

    for i, phi in enumerate(phis):
        x0 = 1.5 * np.cos(phi)
        y0 = 1.5 * np.sin(phi)

        R = np.sqrt((X - x0)**2 + (Y - y0)**2)
        s[i, ...] = 1 / R**2

    # repeat 2D arrays into 3D array
    s = np.repeat(s[:, :, :, np.newaxis], n, axis=3)

    sens = np.zeros((num_channels, n, n, n, 2))
    # generate random phase
    for i, phi in enumerate(phis):
        phase = gaussian_filter(np.random.rand(n, n, n)[:, :, n // 2], n / 6)
        phase -= phase.min()
        phase *= (2 * np.pi / phase.max())

        sens[i, ..., 0] = s[i, ...] * np.cos(phi)
        sens[i, ..., 1] = s[i, ...] * np.sin(phi)

    return sens
