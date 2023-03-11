import numpy as np
import torch
import functions as fcts
import matplotlib.pyplot as plt
from copy import deepcopy

#K1_high = 0.9
#K1_low = 0.6
#
#Vt = 1.0
#fbv = 0.03
#
#k2_high = K1_high / Vt
#k2_low = K1_low / Vt


def generate_random_IF() -> fcts.ExpConvSumFunction:

    while True:
        scale = 0.1 * torch.rand(1)[0]

        p = scale * (10. * torch.rand(1)[0])
        mu1 = 3. * torch.rand(1)[0] + 1
        a1 = scale * (50. * torch.rand(1)[0])
        mu2 = 7. * torch.rand(1)[0] + 1
        a2 = scale * (-53 * torch.rand(1)[0])
        mu3 = 0.4 * torch.rand(1)[0] + 0.1

        a3 = (-p - a1 - a2)

        g1 = fcts.ExpDecayFunction(mu1)
        g1.scale = a1
        g2 = fcts.ExpDecayFunction(mu2)
        g2.scale = a2
        g3 = fcts.ExpDecayFunction(mu3)
        g3.scale = a3
        p1 = fcts.PlateauFunction()
        p1.scale = p

        # generate an arterial input function as sum of 3 exponentials + a plateau
        C_A = fcts.ExpConvSumFunction([g1, g2, g3, p1])

        t = torch.linspace(0, 10, 10 * 10, dtype=torch.float64)
        y = C_A(t)

        if (y.min() >= 0) and (y[-1] < y[-2]) and ((
            (y[-2] - y[-1]) / (t[-1] - t[-2])) < 0.01 * y.max()):

            break

    return C_A


def generate_random_PET(C_A: fcts.ExpConvFunction, Vt=1.):
    K1 = torch.rand(1)[0] + 0.2
    k2 = K1 / Vt

    fbv = 0.05 * torch.rand(1)[0]

    # calculate tissue response
    C_t = C_A.expconv(k2)
    C_t.scale = K1

    scaled_CA = deepcopy(C_A)
    scaled_CA.scale *= fbv
    tmp = deepcopy(C_t)
    tmp.scale *= (1 - fbv)

    C_PET = fcts.IntegrableSumFunction([scaled_CA, tmp])

    return C_PET


class IF_1TCM_DataSet(torch.utils.data.Dataset):

    def __init__(self, tmax=8, num_t=12 * 8, num_reg=2, dtype=torch.float32):
        self._tmax = tmax
        self._num_t = num_t
        self._t = torch.linspace(0, tmax, num_t, dtype=dtype)
        self._num_reg = num_reg
        self._dtype = dtype

    @property
    def t(self) -> torch.Tensor:
        return self._t

    @property
    def num_reg(self) -> int:
        return self._num_reg

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, idx: int):
        C_A = generate_random_IF()

        c_pet = torch.zeros(self._num_reg, self._num_t, dtype=self._dtype)

        for i in range(self._num_reg):
            c_pet[i, :] = generate_random_PET(C_A)(self._t)

        return c_pet, C_A(self._t)


if __name__ == '__main__':

    batch_size = 5
    ds = IF_1TCM_DataSet()

    data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    while True:
        cp, ca = next(iter(data_loader))

        fig, ax = plt.subplots(1,
                               batch_size,
                               sharey=True,
                               figsize=(batch_size * 3, 3))
        for i in range(batch_size):
            ax[i].plot(ds.t, ca[i, :], '.-')
            for j in range(ds.num_reg):
                ax[i].plot(ds.t, cp[i, j, :], '.-')
        fig.tight_layout()
        fig.show()

        tmp = input("Continue?")
        plt.close(fig)