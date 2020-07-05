import cupy as cp
import math
import numpy as np


def auto_detect(shape: tuple, matmul=False):
    assert isinstance(shape, tuple)
    grid_dim = ()
    block_dim = ()
    if len(shape) == 4:
        raise NotImplementedError
    elif len(shape) == 3:
        # if shape[1] < 16 and shape[2] < 16:
        #     block_dim = (1, shape[1], shape[2])
        # elif shape[1] < 16 and shape[2] >= 16:
        #     block_dim = (1, shape[1], math.floor(256 / shape[1]))
        # elif shape[2] < 16 and shape[1] >= 16:
        #     block_dim = (1, math.floor(256 / shape[2]), shape[2])
        # else:
        #     block_dim = (1, 16, 16)
        block_dim = (1, 16, 16)
        grid_dim = (
        math.ceil(shape[0] / block_dim[0]), math.ceil(shape[1] / block_dim[1]), math.ceil(shape[2] / block_dim[2]))
    elif len(shape) == 2:
        if matmul:
            block_dim = (32, 32)
            grid_edge = max(math.ceil(shape[0] / block_dim[0]), math.ceil(shape[0] / block_dim[1]))
            grid_dim = (grid_edge, grid_edge)
            return grid_dim, block_dim
        if shape[0] < 32 and shape[1] < 32:
            block_dim = (shape[0], shape[1])
            grid_dim = (1, 1)
        elif shape[1] < 32:
            block_dim = (math.floor(1024 / shape[1]), shape[1])
            grid_dim = math.ceil(shape[0] / block_dim[0]), math.ceil(shape[1] / block_dim[1])
        elif shape[0] < 32:
            block_dim = (shape[0], math.floor(1024 / shape[0]))
            grid_dim = math.ceil(shape[0] / block_dim[0]), math.ceil(shape[1] / block_dim[1])
        else:
            block_dim = (32, 32)
            grid_dim = math.ceil(shape[0] / block_dim[0]), math.ceil(shape[1] / block_dim[1])
    elif len(shape) == 1:
        if matmul:
            block_dim = (32, 32)
            grid_dim = (math.ceil(shape[0] / 32), math.ceil(shape[0] / 32))
            return grid_dim, block_dim
        if shape[0] < 1024:
            block_dim = (shape[0], 1)
            grid_dim = (1, 1)
        else:
            block_dim = (1024, 1)
            grid_dim = (1, 1)
    else:
        raise ValueError

    return grid_dim, block_dim


_cu_im2col_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void cu_im2col(float * A, float * B, int output_span, int output_sz, int kernel_sz, int in_c, 
                   int limit_x, int limit_y,
                   int dim_k, int dim_n, int dim_m, int dim_c) {
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= limit_x or y >= limit_y) {
        return;
    }
    int batch_idx = __float2int_rd(fdividef(x,output_span));
    int in_c_idx = y % in_c;
    int kernel_idx = __float2int_rd(fdividef(y,in_c));
    int kernel_idx_y = kernel_idx % kernel_sz;
    int kernel_idx_x = __float2int_rd(fdividef(kernel_idx,kernel_sz));
    int output_idx = x % (output_span);
    int output_idx_y = output_idx % output_sz;
    int output_idx_x = __float2int_rd(fdividef(output_idx,output_sz));
    
    B[x * limit_y + y] = A[batch_idx * dim_n * dim_m * dim_c + (output_idx_x + kernel_idx_x) * dim_m * dim_c + (output_idx_y + kernel_idx_y) * dim_c + in_c_idx];
    }
    ''', 'cu_im2col')


def cu_im2col(A, kernel_shape, from_host=False, to_host=False):
    """
    :param inpt: shape=(m,m,C1,k)
    :return: shape=(c1*a*a,(n+2p-a+1)^2*k)
    """
    if from_host:
        A = cp.array(A)

    batch_sz, h, w, in_c = A.shape
    kernel_sz = kernel_shape[0]
    output_sz = (h - kernel_sz) + 1

    res = cp.zeros(shape=(batch_sz * output_sz * output_sz, kernel_sz * kernel_sz * in_c),
                   dtype=cp.float32)

    output_span = output_sz * output_sz

    grid_dim, block_dim = auto_detect(res.shape)
    _cu_im2col_kernel(grid_dim,
                      block_dim,
                      (A, res, output_span, output_sz, kernel_sz, in_c,
                       res.shape[0], res.shape[1],
                       A.shape[0], A.shape[1], A.shape[2], A.shape[3]))

    if to_host:
        return cp.asnumpy(res)
    else:
        return res


_cu_max_pool_ds_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void cu_max_pool_ds(float * A, float * B, int limit_x, int limit_y, int limit_z, int output_sz, int kernel_size,
                        int dim_k, int dim_n, int dim_m, int dim_c) {

        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int z = blockDim.z * blockIdx.z + threadIdx.z;
        
        if (x >= limit_x or y >= limit_y or z >= limit_z) {
            return;
        }
        int batch_idx = __float2int_rd(fdividef(x,output_sz));
        int output_idx_x = x % output_sz;
        int output_idx_y = y;
        int output_ch = z;
        
        float max_v = - 1 / 0;
        float curr_v = 0.;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                curr_v = A[batch_idx * dim_n * dim_m * dim_c + (output_idx_x * kernel_size + i) * dim_m * dim_c + (output_idx_y * kernel_size + j) * dim_c + output_ch];
                // curr_v = A[x * limit_y * limit_z +  y  * limit_z + z];
                if (curr_v > max_v) {
                    max_v = curr_v;
                }
            }
        }
        
        B[x * limit_y * limit_z +  y  * limit_z + z] = max_v;
        
    }
    ''', 'cu_max_pool_ds')


def cu_max_pool_ds(A, output_shape, kernel_sz, from_host=False, to_host=False):
    if from_host:
        A = cp.array(A)

    batch_sz = A.shape[0]
    output_sz = output_shape[1]

    # A.reshape(A.shape[0] * A.shape[1], A.shape[2], A.shape[3])
    res = cp.zeros(shape=(output_shape[0] * batch_sz, output_shape[1], output_shape[2]), dtype=cp.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_max_pool_ds_kernel(grid_dim,
                           block_dim,
                           (A, res, res.shape[0], res.shape[1], res.shape[2], output_sz, kernel_sz,
                            A.shape[0], A.shape[1], A.shape[2], A.shape[3]))

    res = cp.reshape(res, newshape=(batch_sz, *output_shape))

    if to_host:
        return cp.asnumpy(res)
    else:
        return res


_cu_max_pool_us_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void cu_max_pool_us(float * A, float * B, float * C, int limit_x, int limit_y, int limit_z, int output_sz, int kernel_size,
                        int dim_k, int dim_n, int dim_m, int dim_c) {

        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int z = blockDim.z * blockIdx.z + threadIdx.z;
        
        if (x >= limit_x or y >= limit_y or z >= limit_z) {
            return;
        }
        int batch_idx = __float2int_rd(fdividef(x,output_sz));
        int output_idx_x = x % output_sz;
        int output_idx_y = y;
        int output_ch = z;
        
        float max_v = -1/0;
        int max_idx = - 1;
        int curr_idx = -1;
        float curr_v = 0;

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                curr_idx = batch_idx * dim_n * dim_m * dim_c + (output_idx_x * kernel_size + i) * dim_m * dim_c + (output_idx_y * kernel_size + j) * dim_c + output_ch;
                curr_v = A[curr_idx];
                if (curr_v > max_v) {
                    max_v = curr_v;
                    max_idx = curr_idx;
                }
            }
        }
        
        C[max_idx] = B[x * limit_y * limit_z +  y  * limit_z + z];
        
    }
    ''', 'cu_max_pool_us')


def cu_max_pool_us(A, kernel_sz, last_input, from_host=False, to_host=False):
    if from_host:
        A = cp.array(A)

    output_sz = A.shape[1]
    A = A.reshape(A.shape[0] * A.shape[1], A.shape[2], A.shape[3])
    res = cp.zeros(shape=last_input.shape, dtype=cp.float32)
    grid_dim, block_dim = auto_detect(A.shape)
    _cu_max_pool_us_kernel(grid_dim,
                           block_dim,
                           (last_input, A, res, A.shape[0], A.shape[1], A.shape[2], output_sz, kernel_sz,
                            last_input.shape[0], last_input.shape[1], last_input.shape[2], last_input.shape[3]))

    if to_host:
        return cp.asnumpy(res)
    else:
        return res


_cu_avg_pool_ds_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void cu_avg_pool_ds(float * A, float * B, int limit_x, int limit_y, int limit_z, int output_sz, int kernel_size, int span,
                        int dim_k, int dim_n, int dim_m, int dim_c) {

        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int z = blockDim.z * blockIdx.z + threadIdx.z;

        if (x >= limit_x or y >= limit_y or z >= limit_z) {
            return;
        }
        int batch_idx = __float2int_rd(fdividef(x,output_sz));
        int output_idx_x = x % output_sz;
        int output_idx_y = y;
        int output_ch = z;

        float sum_v = 0;
        float curr_v = 0.;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                sum_v += A[batch_idx * dim_n * dim_m * dim_c + (output_idx_x * kernel_size + i) * dim_m * dim_c + (output_idx_y * kernel_size + j) * dim_c + output_ch];
            }
        }

        B[x * limit_y * limit_z +  y  * limit_z + z] = sum_v / span;

    }
    ''', 'cu_avg_pool_ds')


def cu_avg_pool_ds(A, output_shape, kernel_sz, from_host=False, to_host=False):
    if from_host:
        A = cp.array(A)

    batch_sz = A.shape[0]
    output_sz = output_shape[1]
    span = kernel_sz * kernel_sz
    res = cp.zeros(shape=(output_shape[0] * batch_sz, output_shape[1], output_shape[2]), dtype=cp.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_avg_pool_ds_kernel(grid_dim,
                           block_dim,
                           (A, res, res.shape[0], res.shape[1], res.shape[2], output_sz, kernel_sz, span,
                            A.shape[0], A.shape[1], A.shape[2], A.shape[3]))

    res = cp.reshape(res, newshape=(batch_sz, *output_shape))

    if to_host:
        return cp.asnumpy(res)
    else:
        return res


_cu_avg_pool_us_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void cu_avg_pool_us(float * A, float * B, float * C, int limit_x, int limit_y, int limit_z, int output_sz, int kernel_size, int span,
                        int dim_k, int dim_n, int dim_m, int dim_c) {

        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int z = blockDim.z * blockIdx.z + threadIdx.z;

        if (x >= limit_x or y >= limit_y or z >= limit_z) {
            return;
        }
        int batch_idx = __float2int_rd(fdividef(x,output_sz));
        int output_idx_x = x % output_sz;
        int output_idx_y = y;
        int output_ch = z;
        int raw_idx = x * limit_y * limit_z +  y  * limit_z + z;

        int curr_idx = -1;
        
        float raw_v;

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                curr_idx = batch_idx * dim_n * dim_m * dim_c + (output_idx_x * kernel_size + i) * dim_m * dim_c + (output_idx_y * kernel_size + j) * dim_c + output_ch;
                C[curr_idx] =  B[raw_idx] / span;
            }
        }

    }
    ''', 'cu_avg_pool_us')


def cu_avg_pool_us(A, kernel_sz, last_input, from_host=False, to_host=False):
    if from_host:
        A = cp.array(A)

    output_sz = A.shape[1]
    span = kernel_sz * kernel_sz
    A = A.reshape(A.shape[0] * A.shape[1], A.shape[2], A.shape[3])
    res = cp.zeros(shape=last_input.shape, dtype=cp.float32)
    grid_dim, block_dim = auto_detect(A.shape)
    _cu_avg_pool_us_kernel(grid_dim,
                           block_dim,
                           (last_input, A, res, A.shape[0], A.shape[1], A.shape[2], output_sz, kernel_sz, span,
                            last_input.shape[0], last_input.shape[1], last_input.shape[2], last_input.shape[3]))

    if to_host:
        return cp.asnumpy(res)
    else:
        return res
