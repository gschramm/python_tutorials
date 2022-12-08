import numpy as np
import numpy.typing as npt


def radial_2d_golden_angle(num_spokes: int,
                           num_samples_per_spoke: int,
                           kmax: float = np.pi,
                           mode: str = 'half-spoke',
                           golden_angle: None | float = None) -> npt.NDArray:

    if mode == 'half-spoke':
        if golden_angle is None:
            golden_angle = np.pi * 137.51 / 180
        k1d = np.linspace(0, kmax, num_samples_per_spoke, endpoint=True)
    elif mode == 'full-spoke':
        if golden_angle is None:
            golden_angle = np.pi * 111.25 / 180
        k1d = np.linspace(-kmax, kmax, num_samples_per_spoke, endpoint=False)
    else:
        raise ValueError

    spoke_angles = (np.arange(num_spokes) * golden_angle) % (2 * np.pi)

    k = np.zeros((num_spokes * num_samples_per_spoke, 2))

    for i, angle in enumerate(spoke_angles):
        k[(i * num_samples_per_spoke):((i + 1) * num_samples_per_spoke),
          0] = np.cos(angle) * k1d
        k[(i * num_samples_per_spoke):((i + 1) * num_samples_per_spoke),
          1] = np.sin(angle) * k1d

    return k
