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


def stack_of_2d_golden_angle(num_stacks: int, kzmax: float = np.pi, **kwargs):

    star_kspace_sample_points = radial_2d_golden_angle(**kwargs)
    num_star_samples = star_kspace_sample_points.shape[0]

    kz1d = np.linspace(-kzmax, kzmax, num_stacks, endpoint=False)

    k = np.zeros((num_stacks * num_star_samples, 3))

    for i, kz in enumerate(kz1d):
        start = i * num_star_samples
        end = (i + 1) * num_star_samples

        k[start:end, 0] = kz
        k[start:end, 1:] = star_kspace_sample_points

    return k
