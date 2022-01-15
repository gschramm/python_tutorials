# TODO: (1) understand 0.5 extra shift, (2) bilin. interp in 3D arrays -> all directions, (3) residual in tex interp

import cupy as cp
from cupyx.scipy.ndimage.interpolation import shift


source=r'''
extern "C"{
__global__ void interpKernel(float* output,
                           cudaTextureObject_t texObj,
                           int dim0, int dim1, float d0, float d1)
{
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // Read from texture and write to global memory
    // It seems that in Linear Filtering mode, we always have to add an extra .5f to get the "convential" coordinates
    if (i0 < dim0 && i1 < dim1)
      output[i1 * dim0 + i0] = tex2D<float>(texObj, i0 + d0 + .5f, i1 + d1 + .5f);
}
}
'''

#------------------------------------------------------------------------------------------------------------------

# dimension of the 2D array
dim0  = 6
dim1  = 4

# floating point shift for interpolation
d0 = 0.0
d1 = 0.1

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
output   = cp.zeros_like(tex_data)
expected_output = cp.zeros_like(tex_data)
arr.copy_from(tex_data)

# get the kernel, which copies from texture memory
ker = cp.RawKernel(source, 'interpKernel')

# launch it
block_x = 4
block_y = 4
grid_x = (dim0 + block_x - 1)//block_x 
grid_y = (dim1 + block_y - 1)//block_y
ker((grid_x, grid_y), (block_x, block_y), (output, texobj, dim0, dim1, cp.float32(d0), cp.float32(d1)))

print('\ninput data\n')
print(tex_data)
print('\ntexture interpolation\n')
print(output)
print('\nscipy bilinear interpolation\n')
print(shift(tex_data, (-d1,-d0), order = 1))
