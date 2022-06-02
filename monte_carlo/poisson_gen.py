# tutorial on how to generate Poisson random numbers using the "direct method"
# if we have a sequence of random numbers E following an exp. distribution lam * exp(-x/lam)
# than the integer values where the cumulative sum \sum_{k=1}^n E_k > 1 follows a Poisson
# distribution (homogeneous Poisson point process)

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import factorial

def poisson_generator(lam):
  exp_rand_sum     = 0
  poisson_rand_num = 0
  
  while True:
    # draw exponential random number with exponent 1/lambda
    exp_rand_num = np.random.exponential(scale = 1./lam, size = 1)[0]
    # add to the sum
    exp_rand_sum += exp_rand_num
  
    if exp_rand_sum <= 1:
      poisson_rand_num += 1
    else:
      break

  return poisson_rand_num

def poisson_distribution(lam, n):
  return np.exp(-lam)*(lam**n) / factorial(n)

#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

np.random.seed(1)

# Poisson rate
lam  = 10.5
# number of poisson random variables to simulate
n = 100000

# generatate poisson random variables
poisson_random_variables = np.zeros(n, dtype = np.uint64)

for i in range(n):
  print(i, end = '\r')
  poisson_random_variables[i] = poisson_generator(lam)

#--------------------------------------------------------------------------------------

# plot the histogram
bin_edges   = np.arange(poisson_random_variables.max() + 2) - 0.5
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:]) 

fig, ax = plt.subplots()
# plot the histogram of our poisson random variables
ax.hist(poisson_random_variables, bins = bin_edges, density = True)
# add a plot of the true Poisson distribution
ax.plot(bin_centers, poisson_distribution(lam, bin_centers), 'ro')

fig.tight_layout()
fig.show()
