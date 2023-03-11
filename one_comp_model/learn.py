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


if __name__ == '__main__':
    tt = torch.linspace(0, 8, 8 * 12, dtype=torch.float64)

    while True:
        # generate random input function
        C_A1 = generate_random_IF()
        C_PET1 = generate_random_PET(C_A1)
        C_PET2 = generate_random_PET(C_A1)
        C_PET3 = generate_random_PET(C_A1)

        fig, ax = plt.subplots()
        ax.plot(tt, C_A1(tt), '.-')
        ax.plot(tt, C_PET1(tt), '.-')
        ax.plot(tt, C_PET2(tt), '.-')
        ax.plot(tt, C_PET3(tt), '.-')
        fig.show()

        tmp = input("Continue?")
        plt.close(fig)