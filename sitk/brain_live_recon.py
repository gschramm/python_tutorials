import h5py
import numpy as np
import SimpleITK as sitk

from pathlib import Path
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from time import time

import argparse

def calc_origin(shape, voxsize):
  """ calculate the origin of a volume such that the center is at (0,0,0)
  """
  return 0.5*voxsize*(1 - np.array(shape))

def numpy_array_to_itk_image(vol, voxsize, origin = None):
  image = sitk.GetImageFromArray(np.swapaxes(vol, 0, 2))
  image.SetSpacing(tuple(voxsize.astype(np.float64)))

  if origin is None:
    origin = calc_origin(vol.shape, voxsize) 

  image.SetOrigin(tuple(origin))

  return image

def sitk_image_to_numpy_volume(image):
  vol = np.swapaxes(sitk.GetArrayFromImage(image), 0, 2)
  return vol

#---------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('h5file', help = 'h5 file containing the live recons + parameters')
args = parser.parse_args()

lr      = 1.
sf      = 0.1
metric  = 'MI' # MI or SOS
sm_fwhm           = 6.
mask_slice_offset = 10

with h5py.File(args.h5file, 'r') as data:
  live_recons      = data['frameImages'][:]
  voxsize          = data['voxelsize'][:]
  ref_image        = data['refImageUnmasked'][:]
  ref_image_masked = data['refImage'][:]

  # lm duetto sum of squares parameters
  lmd_params = data['lmDuetto/parms'][:]

live_recons      = np.swapaxes(np.swapaxes(live_recons,1,3),1,2)
ref_image        = np.swapaxes(np.swapaxes(ref_image,0,2),0,1)
ref_image_masked = np.swapaxes(np.swapaxes(ref_image_masked,0,2),0,1)

live_recons_smoothed = np.zeros_like(live_recons)

# post smooth images
for i in range(live_recons.shape[0]):
  live_recons_smoothed[i,...] = gaussian_filter(live_recons[i,...], sm_fwhm/(2.35*voxsize))

aligned_recons   = np.zeros_like(live_recons)
transform_params = np.zeros((live_recons.shape[0],6)) 
reg_times        = np.zeros(live_recons.shape[0])

#---------------------------------------------------------------------------------------

# in the fixed image the origin is different since slices at the bottom have been omitted

origin_moving = calc_origin(live_recons[0,...].shape, voxsize)
origin_fixed  = origin_moving + np.array([0,0,mask_slice_offset*voxsize[2]])

fixed_image = numpy_array_to_itk_image(ref_image_masked, voxsize, origin = origin_fixed)


#for i in range(live_recons.shape[0]):
for i in range(live_recons.shape[0]):
  print(i, live_recons.shape[0])
  moving_image = numpy_array_to_itk_image(live_recons_smoothed[i,...], voxsize, 
                                          origin = origin_moving)
  
  
  # Initial Alignment
  initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                sitk.Euler3DTransform((0.,0.,0.)), 
                sitk.CenteredTransformInitializerFilter.GEOMETRY)
  
  
  # registration
  registration_method = sitk.ImageRegistrationMethod()

  if metric == 'MI':
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins = 50)
  elif metric == 'SOS':
    registration_method.SetMetricAsMeanSquares()
  else:
    raise ValueError('unknown metric')
  
  registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
  registration_method.SetMetricSamplingPercentage(sf)
  
  registration_method.SetInterpolator(sitk.sitkLinear)
  
  # Optimizer settings.
  registration_method.SetOptimizerAsGradientDescentLineSearch( learningRate = lr,
      numberOfIterations = 200, convergenceMinimumValue = 1e-6, convergenceWindowSize = 10)
  registration_method.SetOptimizerScalesFromPhysicalShift()
  
  # Setup for the multi-resolution framework.
  registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
  registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2, 1, 0])
  registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
  
  # Don't optimize in-place, we would possibly like to run this cell multiple times.
  registration_method.SetInitialTransform(initial_transform, inPlace = False)
 
  t0 = time() 
  final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                sitk.Cast(moving_image, sitk.sitkFloat32))
  t1 = time()
  reg_times[i] = t1 - t0

  transform_params[i,:] = final_transform.GetParameters()
  #aligned_recons[i,...] = sitk_image_to_numpy_volume(sitk.Resample(moving_image, 
  #                                               fixed_image, 
  #                                               final_transform, sitk.sitkLinear, 0.0, 
  #                                               moving_image.GetPixelID()))


  # Post registration analysis
  print(f"{i:03} Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}")

#-------------
#-------------
#-------------

fig, ax = plt.subplots(6,1, figsize = (15,9))
for i in range(6):
  ax[i].plot(-transform_params[:,i], label = f'ITK {metric}')
  ax[i].plot(lmd_params[:,i], label = 'LMD SOS')
  ax[i].grid(ls = ':')
ax[0].legend(ncol=1)
fig.tight_layout()
fig.show()
