# tutotial on how to use scipy's sparse matrix in COOrdinate format (coo_matrix)
# to multiply a sparse and dense TOF sinograms followed by summing along the TOF direction

import numpy as np
from scipy.sparse import coo_matrix

np.random.seed(1)

# number of geometrical LORs
nLORs    = 10000
# number of TOF bins
nTOFbins = 27

# create random sparse emission sinogram as normal array
# first dimension is number of geometrical LOR
# second dimension is TOF bin number
print('creating data')
sino1 = np.random.poisson(np.ones((nLORs, nTOFbins))/6)

# create a second non-sparse float sinogram (expectation of sth)
sino2 = np.random.rand(nLORs, nTOFbins)

# create sparse coordinate matrice from sparse emission sino
# this could be also easily created from a LM file like
# coo_matrix(([1,1,1], ([4,3,1],[4,3,0])), shape = (nLORs,nTOFbins))
# if the same events occur multiple times in the event list, coo_matrix will sum the entries
s_sino1 = coo_matrix(sino1)

#------------------------------------------------------------------------------

print('multiplying data')

# multiply both "regular" sinograms
prod   = sino1*sino2
# multiply dense and sparse sinogram
s_prod = s_sino1.multiply(sino2)

# check if products are the same
assert np.all(s_prod.toarray() == prod)

# sum the product of the 2 TOF sinos along the TOF direction
tofsum1 = prod.sum(axis = 1)
tofsum2 = np.array(s_prod.sum(axis = 1)).ravel()

# check if sums are close (up to floating point precision)
assert np.allclose(tofsum1, tofsum2)
