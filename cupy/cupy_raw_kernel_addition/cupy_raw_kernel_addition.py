# minimal example showing how to use raw (external) CUDA kernels with cupy
#
# Aim: unerstand how to load and execute a raw kernel based on addition of two arrays

import cupy as cp
import numpy as np
import math
from functools import reduce

# load a kernel defined in a external file
with open('add_kernel.cu','r') as f:
  add_kernel = cp.RawKernel(f.read(), 'my_add')

#-------------------------------------------------------------

cp.random.seed(1)

# shape of random arrays
shape = (55,55,55)
# number of elemnts of arrays
n     = reduce(lambda x,y: x*y, shape)

# define number of threads per block and and calculate number of blocks per grid
threads_per_block = 64
blocks_per_grid   = math.ceil(n/threads_per_block) 

# define two random device arrays
xd = cp.random.rand(*shape).astype(cp.float32)
yd = cp.random.rand(*shape).astype(cp.float32)
# device array for output
zd  = cp.zeros(shape, dtype=cp.float32)

# execute the kernel
add_kernel((blocks_per_grid,), (threads_per_block,), (xd, yd, zd, n))  # grid, block and arguments

# print first 5 elements
print(xd.ravel()[:5])
print(yd.ravel()[:5])
print(zd.ravel()[:5])

# check against numpy addition
assert np.allclose(cp.asnumpy(zd), cp.asnumpy(xd) + cp.asnumpy(yd))
