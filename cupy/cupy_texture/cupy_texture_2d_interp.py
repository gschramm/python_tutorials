# TODO: (1) understand 0.5 extra shift, (2) bilin. interp in 3D arrays -> all directions, (3) residual in tex interp

import cupy as cp
from cupyx.scipy.ndimage.interpolation import map_coordinates


source=r'''
extern "C"{
__global__ void interpKernel(float* output,
                           cudaTextureObject_t texObj,
                           int dim0, int dim1, float* x0, float* x1, int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) 
    {
      // It seems that in Linear Filtering mode, we always have to add an extra .5f to get the "convential" coordinates
      if (x0[i] < dim0 && x1[i] < dim1)
      {
        output[i] = tex2D<float>(texObj, x0[i] + .5f, x1[i] + .5f);
      }
    }
}
}
'''

#------------------------------------------------------------------------------------------------------------------

cp.random.seed(1)

# dimension of the 2D array
dim0  = 200
dim1  = 180

# number of random points to sample
n = 1000

# set up a texture object
ch  = cp.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
arr = cp.cuda.texture.CUDAarray(ch, dim0, dim1)
res = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arr)
tex = cp.cuda.texture.TextureDescriptor((cp.cuda.runtime.cudaAddressModeClamp, cp.cuda.runtime.cudaAddressModeClamp),
                                         cp.cuda.runtime.cudaFilterModeLinear,
                                         cp.cuda.runtime.cudaReadModeElementType)
texobj = cp.cuda.texture.TextureObject(res, tex)

# allocate input/output arrays
tex_data = cp.arange(1, dim0*dim1 + 1, dtype=cp.float32).reshape(dim1, dim0)
arr.copy_from(tex_data)
 
output = cp.zeros(n, dtype = cp.float32)
x0     = cp.floor((dim0*cp.random.rand(n)).astype(cp.float32)) + (17/256)
x1     = cp.floor((dim1*cp.random.rand(n)).astype(cp.float32)) + (87/256)   

# get the kernel, which copies from texture memory
ker = cp.RawKernel(source, 'interpKernel')

# launch it
threads_per_block = 32
blocks_per_grid   = (n + threads_per_block - 1)//threads_per_block

#ker((blocks_per_grid,), (threads_per_block,), (output, texobj, dim0, dim1, x0, x1, n))
ker((blocks_per_grid,), (threads_per_block,), (output, texobj, dim0, dim1, x0, x1, n))

print('\ninput data\n')
print(tex_data)
print('\ntexture interpolation\n')
print(output)
print('\nscipy bilinear interpolation\n')
print(map_coordinates(tex_data, cp.array([x1,x0]), order = 1))
