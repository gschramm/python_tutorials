# minimal example showing how to use raw (external) CUDA kernels with cupy
#
# Aim: unerstand how to load and execute a raw kernel based on addition of two arrays

import cupy as cp

# load a kernel defined in a external file
with open('joseph3d_fwd_cuda.cu','r') as f:
  proj_kernel = cp.RawKernel(f.read(), 'joseph3d_fwd_cuda_kernel')

#-------------------------------------------------------------
