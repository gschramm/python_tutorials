# script to test the parallelized gradient / divergence from pymirc

import numpy as np
import pymirc.image_operations as pi

# seed the random generator
np.random.seed(1)

# create a random 3D/4D image
shape = (6,200,190,180)

# create random array and pad with 0s
x = np.pad(np.random.rand(*shape), 1)

# allocate array for the gradient
grad_x = np.zeros((x.ndim,) + x.shape, dtype = x.dtype)

# calculate the gradient
pi.grad(x, grad_x) 

# setup random array in gradient space
y = np.random.rand(*grad_x.shape)

# calucate divergence 
div_y = pi.div(y)

# check if operators are adjoint
print(-(x*div_y).sum() / (grad_x*y).sum())
