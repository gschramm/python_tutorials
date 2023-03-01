import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def logis(r, x, n=1000, x0=0.5):

    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = r * x[i] * (1 - x[i])


if __name__ == '__main__':
    r = 3.57
    n = 4096
    x = np.zeros(n)

    logis(r, x, n=n)
    spec = np.fft.fft(x[-n // 2:] - x[-n // 2:].mean())

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey='row')
    ax[0].plot(x)
    ax[1].plot(x)
    ax[2].plot(x)
    ax[1].set_xlim(-0.5, 30)
    ax[2].set_xlim(n - 100.5, n + 0.5)
    fig.tight_layout()
    fig.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(np.fft.fftshift(np.fft.fftfreq(spec.size)),
             np.fft.fftshift(np.abs(spec))**0.3)
    fig2.tight_layout()
    fig2.show()