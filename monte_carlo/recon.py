import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from utils import RotationBased2DProjector, PETAcquisitionModel, OS_MLEM, test_images

nreal      = 100
niter      = 5
ps_FWHM_mm = 6.5

plt.rcParams['image.cmap'] = 'Greys'
np.random.seed(1)

# setup the forward projector
proj = RotationBased2DProjector(100, pix_size_mm = 4, num_subsets = 20)

em_img, att_img = test_images(proj.num_pix)
em_img[em_img > 1] = 1

acq_model = PETAcquisitionModel(proj, att_img)

noise_free_data = acq_model.forward(em_img) 


# run reconstructions
recons    = np.zeros((nreal,) + em_img.shape)
recons_sm = np.zeros((nreal,) + em_img.shape)

# post-smoothing kernel
ps_sig = ps_FWHM_mm  / (2.35 * proj.pix_size_mm)

for i in range(nreal):
  print(f'recon {i:03}', end = '\r')
  noisy_data = np.random.poisson(noise_free_data)
  reconstructor = OS_MLEM(noisy_data, acq_model)
  reconstructor.run(niter)
  recons[i,...] = reconstructor.image.copy()

for i in range(nreal):
  recons_sm[i,...] = gaussian_filter(recons[i,...], ps_sig)

recon_mean = recons.mean(axis = 0)
recon_var  = recons.var(axis = 0, ddof = 1)

recon_sm_mean = recons_sm.mean(axis = 0)
recon_sm_var  = recons_sm.var(axis = 0, ddof = 1)

# plots
fig, ax = plt.subplots(2,3, figsize = (9,6))
ax[0,0].imshow(recons[0,...], vmin = 0 , vmax = 1.5*em_img.max())
ax[0,1].imshow(recon_mean, vmin = 0 , vmax = 1.5*em_img.max())
ax[0,2].imshow(recon_var)
ax[1,0].imshow(recons_sm[0,...], vmin = 0 , vmax = 1.5*em_img.max())
ax[1,1].imshow(recon_sm_mean, vmin = 0 , vmax = 1.5*em_img.max())
ax[1,2].imshow(recon_sm_var)
fig.tight_layout()
fig.show()

# calculate voxel correlations
delta       = np.arange(-15,15)
vox_corr    = np.zeros(delta.shape[0])
vox_corr_sm = np.zeros(delta.shape[0])
for i, d in enumerate(delta):
  vox_corr[i]    = np.corrcoef(recons[:,50,50], recons[:,50,50+d])[0,1]
  vox_corr_sm[i] = np.corrcoef(recons_sm[:,50,50], recons_sm[:,50,50+d])[0,1]

fig2, ax2   = plt.subplots()
ax2.plot(delta, vox_corr, '-', marker = '.')
ax2.plot(delta, vox_corr_sm, '-', marker = '.')
ax2.grid(ls = ':')
fig2.tight_layout()
fig2.show()
