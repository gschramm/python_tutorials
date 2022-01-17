# TODO: (1) understand 0.5 extra shift, (2) bilin. interp in 3D arrays -> all directions, (3) residual in tex interp

import cupy as cp
from cupyx.scipy.ndimage.interpolation import shift


source=r'''
extern "C"{
__global__ void interpKernel(float* output,
                           cudaTextureObject_t texObj,
                           int dim0, int dim1, int dim2, float d0, float d1, float d2)
{
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i2 = blockIdx.z * blockDim.z + threadIdx.z;

    // Read from texture and write to global memory
    // It seems that in Linear Filtering mode, we always have to add an extra .5f to get the "convential" coordinates
    if (i0 < dim0 && i1 < dim1 && i2 < dim2)
      output[i2*dim1*dim0 + i1*dim0 + i0] = tex3D<float>(texObj, i0 + d0 + .5f, i1 + d1 + .5f, i2 + d2 + .5f);
}
}
'''

#------------------------------------------------------------------------------------------------------------------

# dimension of the 2D array
dim0  = 300
dim1  = 380
dim2  = 320

# floating point shift for interpolation
d0 = 0.1
d1 = 0.1
d2 = 0.1

# set up a texture object
ch  = cp.cuda.texture.ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
arr = cp.cuda.texture.CUDAarray(ch, dim0, dim1, dim2)
res = cp.cuda.texture.ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=arr)
tex = cp.cuda.texture.TextureDescriptor((cp.cuda.runtime.cudaAddressModeClamp, cp.cuda.runtime.cudaAddressModeClamp),
                                         cp.cuda.runtime.cudaFilterModeLinear,
                                         cp.cuda.runtime.cudaReadModeElementType)
texobj = cp.cuda.texture.TextureObject(res, tex)

# allocate input/output arrays
tex_data = cp.arange(1, dim0*dim1*dim2 + 1, dtype=cp.float32).reshape(dim2, dim1, dim0)
output   = cp.zeros_like(tex_data)
expected_output = cp.zeros_like(tex_data)
arr.copy_from(tex_data)

# get the kernel, which copies from texture memory
ker = cp.RawKernel(source, 'interpKernel')

# launch it
block_x = 4
block_y = 4
block_z = 4
grid_x = (dim0 + block_x - 1)//block_x 
grid_y = (dim1 + block_y - 1)//block_y
grid_z = (dim2 + block_z - 1)//block_z
ker((grid_x, grid_y, grid_z), (block_x, block_y, block_z), (output, texobj, dim0, dim1, dim2, cp.float32(d0), cp.float32(d1), cp.float32(d2)))

print('\ninput data\n')
print(tex_data)
print('\ntexture interpolation\n')
print(output)
print('\nscipy bilinear interpolation\n')
print(shift(tex_data, (-d2,-d1,-d0), order = 1))
