# demo script that shows how to use simple ITK to align a CT and "simulated" noisy and low res PET image
# to each other using mutual information

# Notes:
# - GradientDescent with LineSearch seems to work better than pure GradientDescent
# - learning rate < 1 seems more stable
# - random sampling fraction around 0.1 seems reasonable to balance speed and accuracy

import SimpleITK as sitk
import pymirc.viewer as pv
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path

# input parameters

rel_noise_level = 6.    # relative noise level of the simulated PET image
fwhm_mm         = 6.    # resolution of the simulated PET image
lr              = 0.5   # learning rate / step size of the Gradient-based optimizer
random_sampling_fraction = 0.1

#---------------------------------------------------------------------------------------------------

np.random.seed(1)

fixed_image          = sitk.ReadImage(str(Path('data') / 'BraTS20_Training_009' / 'BraTS20_Training_009_t2.nii'), sitk.sitkFloat32)
moving_image_perfect = sitk.ReadImage(str(Path('data') / 'BraTS20_Training_009' / 'BraTS20_Training_009_t1ce.nii'), sitk.sitkFloat32)

# transform the moving image
true_transform = sitk.Euler3DTransform((119.5, -119.5, 77), 0.5, -0.4, 0.3, (10, -15, 5))

moving_image_perfect = sitk.Resample(image1 = moving_image_perfect, 
                                     transform = true_transform.GetInverse(), 
                                     interpolator = sitk.sitkLinear, 
                                     defaultPixelValue = 0)


# simulate a PET image by adding Poisson noise to the MR and smoothing
tmp = sitk.GetArrayFromImage(moving_image_perfect)

if rel_noise_level > 0:
  scale = tmp.max() * rel_noise_level
  tmp   = np.random.poisson(tmp/scale).astype(np.float32) * scale
  # all also a bit of gaussian noise to have noise in the background
  tmp  += 0.1*scale*np.random.randn(*tmp.shape)

if fwhm_mm > 0:
  tmp = gaussian_filter(tmp, fwhm_mm / (2.35*np.array(moving_image_perfect.GetSpacing())[::-1]))

moving_image = sitk.GetImageFromArray(tmp)
moving_image.SetOrigin(moving_image_perfect.GetOrigin())
moving_image.SetSpacing(moving_image_perfect.GetSpacing())
moving_image.SetDirection(moving_image_perfect.GetDirection())

# Initial Alignment
initial_transform = sitk.CenteredTransformInitializer(
    fixed_image,
    moving_image,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY,
)

moving_resampled = sitk.Resample(
    moving_image,
    fixed_image,
    initial_transform,
    sitk.sitkLinear,
    0.0,
    moving_image.GetPixelID(),
)

moving_perfect_resampled = sitk.Resample(
    moving_image_perfect,
    fixed_image,
    initial_transform,
    sitk.sitkLinear,
    0.0,
    moving_image.GetPixelID(),
)

a = np.swapaxes(sitk.GetArrayFromImage(fixed_image),0,2)
b = np.swapaxes(sitk.GetArrayFromImage(moving_resampled),0,2)
c = np.swapaxes(sitk.GetArrayFromImage(moving_perfect_resampled),0,2)
vi = pv.ThreeAxisViewer([a,b,c,a],[None,None,None,c], voxsize = fixed_image.GetSpacing())



# Registration

registration_method = sitk.ImageRegistrationMethod()

# Similarity metric settings.
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(random_sampling_fraction)

registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer settings.
#registration_method.SetOptimizerAsGradientDescent(
registration_method.SetOptimizerAsGradientDescentLineSearch(
    learningRate=lr,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10,
)

registration_method.SetOptimizerScalesFromPhysicalShift()

# Setup for the multi-resolution framework.
registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Don't optimize in-place, we would possibly like to run this cell multiple times.
registration_method.SetInitialTransform(initial_transform, inPlace=False)

final_transform = registration_method.Execute(
    sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32)
)


# Post registration analysis

print(f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}")
print(f"Final metric value: {registration_method.GetMetricValue()}")
print(f"Final parameters .: {final_transform.GetParameters()}")
print(f"True parameters ..: {true_transform.GetParameters()}")

moving_resampled = sitk.Resample(
    moving_image,
    fixed_image,
    final_transform,
    sitk.sitkLinear,
    0.0,
    moving_image.GetPixelID(),
)

moving_perfect_resampled = sitk.Resample(
    moving_image_perfect,
    fixed_image,
    final_transform,
    sitk.sitkLinear,
    0.0,
    moving_image.GetPixelID(),
)
# show final result

a = np.swapaxes(sitk.GetArrayFromImage(fixed_image),0,2)
b = np.swapaxes(sitk.GetArrayFromImage(moving_resampled),0,2)
c = np.swapaxes(sitk.GetArrayFromImage(moving_perfect_resampled),0,2)
vi2 = pv.ThreeAxisViewer([a,b,c,a],[None,None,None,c], voxsize = fixed_image.GetSpacing())
