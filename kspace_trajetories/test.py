"""script to understand sampliong of fourier space and discrete FT better"""
import abc
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from functions import SquareSignal, TriangleSignal, GaussSignal, CompoundAnalysticalFourierSignal, SquaredL2Norm, L2L1Norm
from operators import FFT, GradientOperator, T2CorrectedFFT
from algorithms import PDHG


class DiffMetric(abc.ABC):

    def __init__(self, y: npt.NDArray, weights=None, xp=np) -> None:
        self._y = y
        self._weights = None
        self._xp = xp

    @abc.abstractmethod
    def _call_from_diff(self, d: npt.NDArray) -> float:
        raise NotImplementedError

    def _diff(self, x: npt.NDArray) -> npt.NDArray:
        return x - self._y

    def __call__(self, x: npt.NDArray) -> float:
        diff = self._diff(x)
        if self._weights is not None:
            diff *= self._weights

        return self._call_from_diff(diff)


class MSE(DiffMetric):

    def _call_from_diff(self, d: npt.NDArray) -> float:
        return (self._xp.conj(d) * d).sum().real


class MAE(DiffMetric):

    def _call_from_diff(self, d: npt.NDArray) -> float:
        return self._xp.abs(d).sum()


class TPITrajectory:

    def __init__(self, fname: str = 'G16_v1.txt', kmax: float = 1.) -> None:
        self._fname = fname
        self._kmax = kmax

        tmp = np.loadtxt(fname)
        self._t_ms = tmp[:, 1]
        self._k = tmp[:, 0] * self._kmax / tmp[:, 0].max()

        self._t_of_k = interp1d(self._k, self._t_ms)

    def t_of_k(self, k: npt.NDArray, factor: float = 1.) -> npt.NDArray:
        return factor * self._t_of_k(np.abs(k))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_level', default=0.2, type=float)
    parser.add_argument('--gradient_factor', default=1., type=float)
    parser.add_argument('--prior',
                        default='L1L2Norm',
                        choices=['L1L2Norm', 'SquaredL2Norm'])
    args = parser.parse_args()

    xp = np

    n = 64
    x0 = 110
    noise_level = args.noise_level
    num_iter = 4000
    rho = 1e1
    prior = args.prior
    betas = xp.logspace(-1, 2, 15)
    T2star_factor = 1.
    readout_time_factor = 1 / args.gradient_factor
    seed = 2
    model_T2star = True
    verbose = False

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    xp.random.seed(seed)

    signal_csf1 = SquareSignal(stretch=16 / x0,
                               scale=1,
                               shift=29 * x0 / 32,
                               T2star=40 * T2star_factor)
    signal_csf2 = SquareSignal(stretch=16 / x0,
                               scale=1,
                               shift=-29 * x0 / 32,
                               T2star=40 * T2star_factor)
    signal_gm1 = SquareSignal(stretch=4. / x0,
                              scale=0.5,
                              shift=3 * x0 / 4,
                              T2star=9 * T2star_factor)
    signal_gm2 = SquareSignal(stretch=4. / x0,
                              scale=0.5,
                              shift=-3 * x0 / 4,
                              T2star=9 * T2star_factor)
    signal_wm1 = SquareSignal(stretch=2 / x0,
                              scale=0.45,
                              shift=(-3 * x0 / 8),
                              T2star=8 * T2star_factor)
    signal_wm2 = SquareSignal(stretch=2 / x0,
                              scale=0.45,
                              shift=(3 * x0 / 8),
                              T2star=8 * T2star_factor)
    signal_lesion = SquareSignal(stretch=4. / x0,
                                 scale=0.65,
                                 shift=0,
                                 T2star=8 * T2star_factor)
    signal = CompoundAnalysticalFourierSignal([
        signal_csf1, signal_csf2, signal_gm1, signal_gm2, signal_wm1,
        signal_wm2, signal_lesion
    ])

    x, dx = xp.linspace(-x0, x0, n, endpoint=False, retstep=True)

    fft = FFT(x, xp=xp)
    k = fft.k

    tpi_trajectory = TPITrajectory(
        fname='G16_v1.txt', kmax=0.914
    )  # 0.914 is the max k-value for the FFT with x0 = 110 and n = 64

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # generate data from continuous FFT
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    noise_free_data = xp.zeros(n, dtype=xp.complex128)
    t_readout = xp.zeros(n)

    for i, kk in enumerate(k):
        t_r = tpi_trajectory.t_of_k(kk, factor=readout_time_factor)
        t_readout[i] = t_r
        noise_free_data[i] = signal.continous_ft(kk, t=t_r)

    scaled_noise_level = noise_level / xp.sqrt(readout_time_factor)

    data = noise_free_data.copy() + scaled_noise_level * xp.random.randn(
        *noise_free_data.shape) + 1j * noise_level * xp.random.randn(
            *noise_free_data.shape)

    print(f'readout time factor .: {readout_time_factor:.1e}')
    print(f'noise level         .: {noise_level:.1e}')
    print(f'scaled noise level  .: {scaled_noise_level:.1e}')

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # inverse fourier transform reconstruction
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    recon1 = fft.inverse(data)

    if model_T2star:
        data_operator = T2CorrectedFFT(x, t_readout, signal.T2star(x), xp=xp)
    else:
        data_operator = fft

    data_distance = SquaredL2Norm(xp, scale=1.0, shift=data)

    prior_operator = GradientOperator(x.shape, xp=xp)

    data_operator_norm = data_operator.norm(num_iter=200)

    pdhg_recons = xp.zeros((len(betas), n), dtype=xp.complex128)
    pdhg_costs = xp.zeros((len(betas), num_iter), dtype=xp.float64)

    for i, beta in enumerate(betas):
        if verbose:
            print(f'{i+1}/{betas.size}')
        if prior == 'SquaredL2Norm':
            prior_norm = SquaredL2Norm(xp, scale=beta)
        elif prior == 'L1L2Norm':
            prior_norm = L2L1Norm(xp, scale=beta)
        else:
            raise ValueError

        pdhg = PDHG(data_operator=data_operator,
                    data_distance=data_distance,
                    sigma=0.5 * rho / data_operator_norm,
                    tau=0.5 / (rho * data_operator_norm),
                    prior_operator=prior_operator,
                    prior_functional=prior_norm)
        pdhg.run(num_iter, verbose=False, calculate_cost=True)
        pdhg_recons[i, :] = pdhg.x
        pdhg_costs[i, :] = pdhg.cost

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # metrics
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    s_true = signal.signal(x)
    #weights = (s_true.real > 0).astype(xp.float64)
    weights = None
    metrics_dict = dict(MSE=MSE(y=s_true, weights=weights, xp=xp),
                        MAE=MAE(y=s_true, weights=weights, xp=xp))
    results = {}

    for key, metric in metrics_dict.items():
        results[key] = [metric(recon) for recon in pdhg_recons]

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # plots
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    xx = xp.linspace(-x0, x0, 1000, endpoint=False)
    kk = xp.linspace(k.min(), k.max(), 1000, endpoint=False)
    it = xp.arange(1, num_iter + 1)

    fig, ax = plt.subplots(1, 5, figsize=(16, 4))
    ax[0].plot(xx, signal.signal(xx).real, 'k-', lw=0.5)
    ax[1].plot(xx, signal.signal(xx).imag, 'k-', lw=0.5)
    ax[0].plot(x, recon1.real, '-', lw=0.8)
    ax[1].plot(x, recon1.imag, '-', lw=0.8, label='IFFT')
    for i, beta in enumerate(betas):
        ax[0].plot(x, pdhg_recons[i, :].real, '-', lw=0.8)
        ax[1].plot(x,
                   pdhg_recons[i, :].imag,
                   '-',
                   lw=0.8,
                   label=f'{prior} {beta:.1e}',
                   color=plt.cm.tab10((i + 1) % 10))

    ax[2].plot(kk, signal.continous_ft(kk).real, 'k-', lw=0.5)
    ax[2].plot(k, noise_free_data.real, 'x', ms=4)
    ax[2].plot(k, data.real, '.', ms=4)

    for i, beta in enumerate(betas):
        ax[3].loglog(it, pdhg_costs[i, ...], color=plt.cm.tab10((i + 1) % 10))

    for axx in ax.ravel():
        axx.grid(ls=':')
    ax[1].set_ylim(*ax[0].get_ylim())
    ax[1].legend()

    ax[4].plot(k, t_readout, '.')

    fig.tight_layout()
    fig.show()

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # plot metrics
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    fig2, ax2 = plt.subplots(3,
                             len(metrics_dict),
                             figsize=(3 * len(metrics_dict), 3 * 3))
    i = 0
    for key, metric in results.items():
        imin = xp.argmin(metric)
        print(
            f'{key}:   opt.beta: {betas[imin]:.2e}   opt.value: {metric[imin]:.2e}'
        )
        ax2[0, i].loglog(betas, metric, 'x-')
        ax2[0, i].loglog([betas[imin]], [metric[imin]], 'x')
        ax2[0, i].set_xlabel('beta')
        ax2[0, i].set_ylabel(key)
        ax2[1, i].plot(xx, signal.signal(xx).real, 'k-', lw=0.5)
        ax2[1, i].plot(x, pdhg_recons[imin, :].real, '-', lw=0.8)
        ax2[2, i].plot(xx, signal.signal(xx).imag, 'k-', lw=0.5)
        ax2[2, i].plot(x, pdhg_recons[imin, :].imag, '-', lw=0.8)
        ax2[0, i].set_title(key)
        i += 1

    for axx in ax2.ravel():
        axx.grid(ls=':')

    fig2.tight_layout()
    fig2.show()

    fig3, ax3 = plt.subplots(1, 3, figsize=(9, 3))
    ax3[0].plot(xx, signal.signal(xx, t=0).real, '-', lw=0.5)
    ax3[0].plot(xx, signal.signal(xx, t=t_readout.max() / 2).real, '-', lw=0.5)
    ax3[0].plot(xx, signal.signal(xx, t=t_readout.max()).real, '-', lw=0.5)
    ax3[1].plot(xx, signal.signal(xx, t=0).imag, '-', lw=0.5)
    ax3[1].plot(xx, signal.signal(xx, t=t_readout.max() / 2).imag, '-', lw=0.5)
    ax3[1].plot(xx, signal.signal(xx, t=t_readout.max()).imag, '-', lw=0.5)
    ax3[2].plot(xx, signal.T2star(xx), '-', lw=0.5)
    ax3[0].grid(ls=':')
    ax3[1].grid(ls=':')
    fig3.tight_layout()
    fig3.show()
