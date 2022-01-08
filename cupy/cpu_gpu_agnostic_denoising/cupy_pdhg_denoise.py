# demo implementation of PDHG denoiser that can run on CPU and GPU
#
# Aim of this tutorial: understand how to write CPU/GPU agnostic code 
#
# cupy implements almost all of numpy's function for GPU arrays
# see e.g. https://docs.cupy.dev/en/stable/user_guide/basic.html#how-to-write-cpu-gpu-agnostic-code
#
#-------------------------------------------------------------------------------------------------------

# if you want to run everthing on a GPU using cupy, import cupy as xp
# if you don't have a GPU, you can run everything on CPU using numpy as xp

# comment one of them
import cupy  as xp
#import numpy as xp

class GradientOperator:
  """ finite forward difference gradient operator """

  def __init__(self, ndim):
    self.ndim     = ndim
    self.sl_first = slice(0,1,None)
    self.sl_last  = slice(-1,None,None)
    self.sl_all   = slice(None,None,None)

  def fwd(self, x):
    g = xp.zeros((self.ndim,) + x.shape, dtype = x.dtype)

    # we use numpy's/ cupy's diff functions and append/prepend the first/last slice
    for i in range(self.ndim):
      sl = i*(self.sl_all,) + (self.sl_last,) + (self.ndim - i - 1)*(self.sl_all,)
      g[i,...] = xp.diff(x, axis = i, append = x[sl])

    return g

  def adjoint(self, y):
    d = xp.zeros(y[0,...].shape, dtype = y.dtype)
    
    for i in range(self.ndim):
      sl = i*(self.sl_all,) + (self.sl_first,) + (self.ndim - i - 1)*(self.sl_all,)
      d -= xp.diff(y[i,...], axis = i, prepend = y[i,...][sl])

    return d

#-----------------------------------------------------------------------------------

class GradientNorm:
  """ 
  norm of a gradient field

  Parameters
  ----------

  name : str
    name of the norm
    'l2_l1' ... mixed L2/L1 (sum of pointwise Euclidean norms in every voxel)
    'l2_sq' ... squared l2 norm (sum of pointwise squared Euclidean norms in every voxel)
  """
  def __init__(self, name = 'l2_l1'):
    self.name = name
 
    if not self.name in ['l2_l1', 'l2_sq']:
     raise NotImplementedError

  def eval(self, x):
    if self.name == 'l2_l1':
      n = xp.linalg.norm(x, axis = 0).sum()
    elif self.name == 'l2_sq':
      n = (x**2).sum()

    return n

  def prox_convex_dual(self, x, sigma = None):
    """ proximal operator of the convex dual of the norm
    """
    if self.name == 'l2_l1':
      gnorm = xp.linalg.norm(x, axis = 0)
      r = x/xp.clip(gnorm, 1, None)
    elif self.name == 'l2_sq':
      r = x/(1+sigma)

    return r

#-----------------------------------------------------------------------------------

def pdhg_l2_denoise(img, grad_operator, grad_norm, 
                    weights = 2e-2, niter = 200, cost = None, nonneg = False, verbose = False):
  """
  First-order primal dual image denoising with weighted L2 data fidelity term.
  Solves the problem: argmax_x( \sum_i w_i*(x_i - img_i)**2 + norm(grad_operator x) )

  Argumtents
  ----------

  img           ... an nd image image

  grad_operator ... gradient operator with methods fwd() and adjoint()

  grad_norm     ... gradient norm with methods eval() and prox_convex_dual()

  Keyword arguments
  -----------------

  weights  ... (scalar or array) with weights for data fidelity term - default 2e-2

  niter    ... (int) number of iterations to run - default 200

  cost     ... (1d array) 1d output array for cost calcuation - default None
 
  nonneg   ... (bool) whether to clip negative values in solution - default False

  verbose  ... (bool) whether to print some diagnostic output - default False
  """

  x    = img.copy().astype(xp.float32)
  xbar = x.copy()
  
  ynew = xp.zeros((x.ndim,) + x.shape, dtype = x.dtype)

  if weights is xp.array: gam = weights.min()  
  else:                   gam = weights

  tau    = 1./ gam
  sig    = 1./(tau*4.*x.ndim)
  
  # start the iterations
  for i in range(niter):
    if verbose: print(i)

    # (1) fwd model
    ynew += sig*grad_operator.fwd(xbar)
    
    # (2) proximity operator
    ynew = grad_norm.prox_convex_dual(ynew, sigma = sig)
    
    # (3) back model
    xnew = x - tau*grad_operator.adjoint(ynew)
    
    # (4) apply proximity of G
    xnew = (xnew + weights*img*tau) / (1. + weights*tau)
    if nonneg: xnew = xp.clip(xnew, 0, None)  
    
    # (5) calculate the new stepsizes
    theta = 1.0 / xp.sqrt(1 + 2*gam*tau)
    tau   = tau*theta
    sig   = sig/theta 
    
    # (6) update variables
    xbar = xnew + theta*(xnew  - x)
    x    = xnew.copy()
  
    # (0) store cost 
    if cost is not None: 
      cost[i] = 0.5*(weights*(x - img)**2).sum() + grad_norm.eval(grad_operator.fwd(x))
      if verbose: print(cost[i])

  return x

#-----------------------------------------------------------------------------------

xp.random.seed(1)

x  = xp.ones((50,50,50), dtype = xp.float32)
x  = xp.pad(x,25)
x += 0.5
x  = xp.pad(x,25)

x += 1*xp.random.rand(*x.shape).astype(xp.float32)

g  = GradientOperator(x.ndim)
gn = GradientNorm('l2_l1') 

niter = 100
cost  = xp.zeros(niter, dtype = xp.float32)

from time import time
ta = time()
r = pdhg_l2_denoise(x, g, gn, weights = 3, niter = niter, cost = cost, nonneg = False, verbose = False)
tb = time()
print(tb - ta)



#------------------------------------------------------------------------------------
# plots

# if we used cupy, we need to copy the arrays to the host before plotting them
it = xp.arange(1,niter+1)

if xp.__name__ == 'cupy':
  x    = xp.asnumpy(x)
  r    = xp.asnumpy(r)
  cost = xp.asnumpy(cost)
  it   = xp.asnumpy(it)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,3, figsize = (12,4))
ax[0].imshow(x[x.shape[0]//2,...], vmin = 0, vmax = 2, cmap = plt.cm.Greys)
ax[0].set_title('noisy volume', fontsize = 'medium')
ax[1].imshow(r[r.shape[0]//2,...], vmin = 0, vmax = 2, cmap = plt.cm.Greys)
ax[1].set_title('denoised volume', fontsize = 'medium')
ax[2].loglog(it, cost)
ax[2].set_xlabel('iteration')
ax[2].set_ylabel('cost')
ax[2].grid(ls = ':', which = 'both')
fig.tight_layout()
fig.show()
