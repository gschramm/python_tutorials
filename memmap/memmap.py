""" shows how to use numpy's memmap class to deal with arrays that do not fit in memory
    see https://ipython-books.github.io/48-processing-large-numpy-arrays-with-memory-mapping/ """

import numpy as np

map_name  = 'test.dat'
data_type = np.float32
data_shape = (224, 357, 86, 27)

# create memory map "column by column"
f = np.memmap(map_name, dtype=data_type,
              mode='w+', shape=data_shape)

for i in range(data_shape[0]):
    print(f'generating view {(i+1):03}/{data_shape[0]}', end = '\r')
    x = np.random.rand(*data_shape[1:]).astype(np.float32)
    f[i,...] = x

print('')

del f

# "read" memory map
f2 = np.memmap(map_name, dtype=data_type,
               mode='r' ,shape=data_shape)

# read last column of the data
y = f2[-1,...]

# check if it is the same as the last generated data column
print(np.array_equal(y, x))

# the memory map should behave the same as a regular numpy array
# you should be able to use all of numpy's functions

mean_0 = f2[...,0].mean()
std_1 = f2[1,...].std()
