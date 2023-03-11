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

while True:

    p = 10. * torch.rand(1)[0]
    mu1 = 3. * torch.rand(1)[0] + 1
    a1 = 50. * torch.rand(1)[0]
    mu2 = 7. * torch.rand(1)[0] + 1
    a2 = -53 * torch.rand(1)[0]
    mu3 = 0.4 * torch.rand(1)[0] + 0.1
    a3 = -p - a1 - a2

    g11 = fcts.ExpDecayFunction(mu1)
    g11.scale = a1
    g12 = fcts.ExpDecayFunction(mu2)
    g12.scale = a2
    g13 = fcts.ExpDecayFunction(mu3)
    g13.scale = a3
    p1 = fcts.PlateauFunction()
    p1.scale = p

    # generate an arterial input function as sum of 3 exponentials + a plateau
    C_A1 = fcts.ExpConvSumFunction([g11, g12, g13, p1])

    t = torch.linspace(0, 10, 100, dtype=torch.float64)
    y = C_A1(t)

    if (y.min() >= 0) and (y[-1] < y[-2]) and ((
        (y[-2] - y[-1]) / (t[-1] - t[-2])) < 0.01 * y.max()):
        fig, ax = plt.subplots()
        ax.plot(t, C_A1(t))
        fig.show()

        tmp = input("Continue?")
        plt.close(fig)