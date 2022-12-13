"""script to understand sampliong of fourier space and discrete FT better"""
import numpy as np
import matplotlib.pyplot as plt

from functions import SquareSignal, TriangleSignal, GaussSignal, CompoundAnalysticalFourierSignal, SquaredL2Norm, L2L1Norm
from operators import FFT, GradientOperator
from algorithms import PDHG


def t_of_k(k, factor: float = 1.):
    return factor * 40 * np.abs(k) / 0.91391


def mse(x, y):
    diff = x - y
    return (np.conj(diff) * diff).sum().real


def mae(x, y):
    diff = x - y
    return np.abs(diff).sum()


if __name__ == '__main__':

    xp = np
    xp.random.seed(1)

    n = 128
    x0 = 110.
    noise_level = 0.2
    num_iter = 2000
    rho = 1e2
    prior = 'L1L2Norm'
    betas = np.logspace(-1, 1.5, 9)
    #prior = 'SquaredL2Norm'
    #betas = [1e1, 1e2, 1e3]
    T2star_factor = 1.
    readout_time_factor = 1. / 4

    #-------------------------------------------------------------------
    #-------------------------------------------------------------------
    #-------------------------------------------------------------------

    signal_csf1 = SquareSignal(stretch=20. / x0,
                               scale=1,
                               shift=0.725 * x0,
                               T2star=50 * T2star_factor)
    signal_csf2 = SquareSignal(stretch=20. / x0,
                               scale=1,
                               shift=-0.725 * x0,
                               T2star=50 * T2star_factor)
    signal_gm1 = SquareSignal(stretch=5. / x0,
                              scale=0.5,
                              shift=0.6 * x0,
                              T2star=9 * T2star_factor)
    signal_gm2 = SquareSignal(stretch=5. / x0,
                              scale=0.5,
                              shift=-0.6 * x0,
                              T2star=9 * T2star_factor)
    signal_wm = SquareSignal(stretch=1. / x0,
                             scale=0.45,
                             shift=0,
                             T2star=8 * T2star_factor)
    signal_lesion = SquareSignal(stretch=10. / x0,
                                 scale=0.25,
                                 shift=0,
                                 T2star=8 * T2star_factor)
    signal = CompoundAnalysticalFourierSignal([
        signal_csf1, signal_csf2, signal_gm1, signal_gm2, signal_wm,
        signal_lesion
    ])

    #fig, ax = plt.subplots(1,2, figsize=(8,4))
    #ax[0].plot(xx, signal.signal(xx, t = 0))
    #ax[0].plot(xx, signal.signal(xx, t = 10))
    #ax[0].plot(xx, signal.signal(xx, t = 40))
    #fig.tight_layout()
    #fig.show()

    x, dx = xp.linspace(-x0, x0, n, endpoint=False, retstep=True)

    fft = FFT(x)
    k = fft.k

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # generate data from continuous FFT
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    noise_free_data = xp.zeros(n, dtype=xp.complex128)
    t_readout = xp.zeros(n)

    for i, kk in enumerate(k):
        t_r = t_of_k(kk, factor=readout_time_factor)
        t_readout[i] = t_r
        noise_free_data[i] = signal.continous_ft(kk, t=t_r)

    scaled_noise_level = noise_level / np.sqrt(readout_time_factor)

    data = noise_free_data.copy() + scaled_noise_level * xp.random.randn(
        *noise_free_data.shape) + 1j * noise_level * xp.random.randn(
            *noise_free_data.shape)

    print(f'readout time factor .: {readout_time_factor:.1e}')
    print(f'scaled noise level  .: {scaled_noise_level:.1e}')

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # inverse fourier transform reconstruction
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    recon1 = fft.inverse(data)

    data_distance = SquaredL2Norm(xp, scale=1.0, shift=data)

    prior_operator = GradientOperator(x.shape, xp=xp)

    fft_norm = fft.norm(num_iter=200)

    pdhg_recons = np.zeros((len(betas), n), dtype=np.complex128)
    pdhg_costs = np.zeros((len(betas), num_iter), dtype=np.float64)

    for i, beta in enumerate(betas):
        if prior == 'SquaredL2Norm':
            prior_norm = SquaredL2Norm(xp, scale=beta)
        elif prior == 'L1L2Norm':
            prior_norm = L2L1Norm(xp, scale=beta)
        else:
            raise ValueError

        pdhg = PDHG(data_operator=fft,
                    data_distance=data_distance,
                    sigma=0.5 * rho / fft_norm,
                    tau=0.5 / (rho * fft_norm),
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

    metrics_dict = dict(MSE=mse, MAE=mae)
    results = {}

    for key, metric in metrics_dict.items():
        results[key] = [
            metric(recon, signal.signal(x)) for recon in pdhg_recons
        ]

    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------
    # plots
    #-----------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------

    xx = xp.linspace(-x0, x0, 1000, endpoint=False)
    kk = xp.linspace(k.min(), k.max(), 1000, endpoint=False)
    it = np.arange(1, num_iter + 1)

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
                   label=f'{prior} {beta:.1e}')

    ax[2].plot(kk, signal.continous_ft(kk).real, 'k-', lw=0.5)
    ax[2].plot(k, noise_free_data.real, 'x', ms=4)
    ax[2].plot(k, data.real, '.', ms=4)

    for i, beta in enumerate(betas):
        ax[3].loglog(it, pdhg_costs[i, ...], color=plt.cm.tab10(i + 1))

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
        imin = np.argmin(metric)
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