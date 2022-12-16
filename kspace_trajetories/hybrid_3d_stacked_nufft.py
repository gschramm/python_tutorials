import numpy as np
from operators import MultiChannelStackedNonCartesianMRAcquisitionModel

input_shape = (32, 256, 256)
sens_shape = (3, ) + input_shape

x = np.random.rand(*input_shape) + 1j * np.random.rand(*input_shape)
sens = np.random.rand(*sens_shape) + 1j * np.random.rand(*sens_shape)

kspace_sample_points = 2 * np.pi * np.random.rand(50 * 512, 2) - np.pi

data_operator = MultiChannelStackedNonCartesianMRAcquisitionModel(
    input_shape, sens, kspace_sample_points, device_number=0)

data_operator.adjointness_test()