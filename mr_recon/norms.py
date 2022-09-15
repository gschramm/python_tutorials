import abc
import numpy as np
import numpy.typing as npt


class Norm(abc.ABC):
    """abstract base clase for norms where we can calculate the prox of the convex dual"""

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def prox_convex_dual(self, x: npt.NDArray, sigma: float) -> npt.NDArray:
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
        raise NotImplementedError


class ComplexL1L2Norm(Norm):
    """mixed L1-L2 norm of a pseudo-complex gradient field - real and imaginary part are treated separately"""

    def __init__(self) -> None:
        pass

    def __call__(self, x: npt.NDArray) -> float:
        n = np.linalg.norm(x[..., 0], axis=0).sum() + np.linalg.norm(
            x[..., 1], axis=0).sum()

        return n

    def prox_convex_dual(self, x: npt.NDArray, sigma: float) -> npt.NDArray:

        gnorm0 = np.linalg.norm(x[..., 0], axis=0)
        r0 = x[..., 0] / np.clip(gnorm0, 1, None)

        gnorm1 = np.linalg.norm(x[..., 1], axis=0)
        r1 = x[..., 1] / np.clip(gnorm1, 1, None)

        return np.stack([r0, r1], axis=-1)


class L2NormSquared(Norm):
    """squared L2 norm"""

    def __init__(self) -> None:
        pass

    def __call__(self, x: npt.NDArray) -> float:
        n = 0.5 * (x**2).sum()

        return n

    def prox_convex_dual(self, x: npt.NDArray, sigma: float) -> npt.NDArray:

        return x / (1 + sigma)
