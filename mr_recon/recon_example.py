import numpy as np
import numpy.typing as npt

from phantoms import rod_phantom
from operators import MultiChannel3DCartesianMRAcquisitionModel, ComplexGradientOperator
from norms import L2NormSquared, ComplexL1L2Norm
from algorithms import PDHG

import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------
# input parameters

n: int = 64
num_channels: int = 4
noise_level: float = 0.3
seed: int = 0

num_iter: int = 20

data_norm = L2NormSquared()
prior_norm = ComplexL1L2Norm()
beta: float = 1e1

#----------------------------------------------------------------------------------------
np.random.seed(seed)

# setup ground truth image and simulate data
# ground truth image
ph = rod_phantom(n=n)
x_true = np.stack([ph, np.zeros_like(ph)], axis=-1)

# setup coil sensitivities
sens = np.ones((num_channels, n, n, n, 2))

# setup the data model
data_operator = MultiChannel3DCartesianMRAcquisitionModel(
    n, num_channels, sens)

# generate noise-free data
noise_free_data = data_operator.forward(x_true)

noisy_data = noise_free_data + noise_level * x_true.mean() * np.random.randn(
    *data_operator.x_shape)

# setup the prior operator
prior_operator = ComplexGradientOperator(n, 3)

#----------------------------------------------------------------------------------------
# run PDHG reconstruction

sigma = 0.2 / prior_operator.norm()
tau = 0.2 / prior_operator.norm()

reconstructor = PDHG(noisy_data, data_operator, data_norm, prior_operator,
                     prior_norm, beta, sigma, tau)

reconstructor.run(num_iter, calculate_cost=True)

#----------------------------------------------------------------------------------------
# show the results

ims = dict(cmap=plt.cm.Greys_r, vmin=0, vmax=1.2 * x_true.max())

fig, ax = plt.subplots(2, 3, figsize=(9, 6))
ax[0, 0].imshow(x_true[..., n // 2, 0], **ims)
ax[0, 1].imshow(x_true[..., n // 2, 1], **ims)
ax[0, 2].imshow(np.linalg.norm(x_true, axis=-1)[..., n // 2], **ims)
ax[1, 0].imshow(reconstructor.x[..., n // 2, 0], **ims)
ax[1, 1].imshow(reconstructor.x[..., n // 2, 1], **ims)
ax[1, 2].imshow(np.linalg.norm(reconstructor.x, axis=-1)[..., n // 2], **ims)
fig.tight_layout()
fig.show()