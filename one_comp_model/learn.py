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

    def __init__(self,
                 tmax=8,
                 num_t=12 * 8,
                 num_reg=4,
                 nl=0.1,
                 dtype=torch.float32):
        self._tmax = tmax
        self._num_t = num_t
        self._t = torch.linspace(0, tmax, num_t, dtype=dtype)
        self._num_reg = num_reg
        self._nl = nl
        self._dtype = dtype

    @property
    def t(self) -> torch.Tensor:
        return self._t

    @property
    def num_reg(self) -> int:
        return self._num_reg

    @property
    def num_t(self) -> int:
        return self._num_t

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, idx: int):
        C_A = generate_random_IF()

        c_pet = torch.zeros(self._num_reg, self._num_t, dtype=self._dtype)

        for i in range(self._num_reg):
            c_pet[i, :] = generate_random_PET(C_A)(self._t)

        scale = c_pet.max()

        c_pet /= scale

        c_pet += self._nl * torch.randn(c_pet.shape, dtype=self._dtype)

        return c_pet, C_A(self._t).expand(1, -1) / scale


def training_loop(dataloader, model, loss_fn, optimizer, device):

    loss_list = []
    model.train()

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        x_fwd = model.forward(x)

        loss = loss_fn(x_fwd, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(float(loss.cpu().detach().numpy()))

        if i % 30 == 0:
            print(
                f'{(i+1):03} / {(dataloader.dataset.__len__() // dataloader.batch_size):03} loss: {loss_list[-1]:.2E}'
            )

    return loss_list


if __name__ == '__main__':

    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-3
    #device = torch.device("cuda:0")
    device = torch.device("cpu")

    #loss_fct = torch.nn.L1Loss()
    loss_fct = torch.nn.MSELoss()
    #-------------------------------------------------------------------

    ds = IF_1TCM_DataSet()

    data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    # setup simple dense network
    model = torch.nn.Sequential(
        torch.nn.Conv1d(ds.num_reg, 1, kernel_size=1, padding='same'),
        torch.nn.Linear(ds.num_t, 32), torch.nn.Linear(32, 8),
        torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 32),
        torch.nn.Linear(32, ds.num_t))
    model = model.to(device)

    training_loss = []

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in np.arange(num_epochs):
        print(f'epoch {(epoch+1):04} / {num_epochs:04}')
        training_loss += training_loop(data_loader, model, loss_fct, optimizer,
                                       device)

with torch.no_grad():
    cp, ca = next(iter(data_loader))
    pred = model(cp.to(device))

    num_cols = 8
    num_rows = int(np.ceil(batch_size / num_cols))

    fig, ax = plt.subplots(num_rows,
                           num_cols,
                           figsize=(num_cols * 2, num_cols * 2))
    for i in range(batch_size):
        ax.ravel()[i].plot(ds.t, ca[i, 0, :], '.-')
        for j in range(ds.num_reg):
            ax.ravel()[i].plot(ds.t, cp[i, j, :], '.-')
        ax.ravel()[i].plot(ds.t, pred[i, 0, :], 'x-')
    fig.tight_layout()
    fig.show()
