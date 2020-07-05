from numba import cuda, float32, int32
import math
import numpy as np
import time


# tpb = 32
# bpg = 100
# n = bpg * tpb

# @cuda.jit(argtypes=[float32[:,:], float32[:,:], float32[:,:], int32, int32], target='gpu')
# def cu_matmul(A, B, C, tpb, bpg):
#     """
#     :param A: shape(n,m)
#     :param B: shape(m,k)
#     :param C: output, shape(n,k)
#     :return:
#     """
#     sA = cuda.shared.array(shape=(32, 32), dtype=float32)
#     sB = cuda.shared.array(shape=(32, 32), dtype=float32)
#
#     tx = cuda.threadIdx.x
#     ty = cuda.threadIdx.y
#     bx = cuda.blockIdx.x
#     by = cuda.blockIdx.y
#     bw = cuda.blockDim.x
#     bh = cuda.blockDim.y
#
#     x = tx + bx * bw
#     y = ty + by * bh
#
#     acc = 0.
#     for i in range(bpg):
#         if x < tpb * bpg and y < tpb * bpg:
#             sA[ty, tx] = A[y, tx + i * tpb]
#             sB[ty, tx] = B[ty + i * tpb, x]
#
#         cuda.syncthreads()
#
#         if x < tpb * bpg and y < tpb * bpg:
#             for j in range(tpb):
#                 acc += sA[ty, j] * sB[j, tx]
#
#         cuda.syncthreads()
#
#     if x < tpb * bpg and y < tpb * bpg:
#         C[y, x] = acc


def auto_detect(shape: tuple, matmul=False):
    assert isinstance(shape, tuple)
    grid_dim = ()
    block_dim = ()
    if len(shape) == 4:
        raise NotImplementedError
    elif len(shape) == 3:
        if shape[1] < 32:
            block_dim = (math.floor(256 / shape[1]), shape[1])
            grid_dim = math.ceil(shape[0] / block_dim[0]), math.ceil(shape[0] / block_dim[1])
        else:
            block_dim = (32, 32)
            grid_dim = math.ceil(shape[0] / block_dim[0]), math.ceil(shape[0] / block_dim[1])
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


# @cuda.jit()
# def cu_2d_matrix_sum_axis_1(A, B, tpb, bpg):
#     """
#     :param A: input, shape(n,m)
#     :param B: input, shape(n,1)
#     :return:
#     """
#     sA = cuda.shared.array(shape=(32, 32), dtype=float32)
#     acc = cuda.shared.array(shape=32, dtype=float32)
#     tx = cuda.threadIdx.x
#     ty = cuda.threadIdx.y
#
#     x, y = cuda.grid(2)
#
#     for i in range(bpg):
#         if x < A.shape[0] and y < A.shape[1]:
#             sA[ty, tx] = A[y, tx + i * tpb]
#         cuda.syncthreads()
#
#         if x < C.shape[0] and y < C.shape[1]:
#             for j in range(tpb):
#                 for k in range(tpb):
#                     acc[j] += sA[j, k]
#         cuda.syncthreads()
#
#     for i in range


@cuda.jit
def _cu_matmul(A, B, C):
    """
    :param A:
    :param B:
    :param C: C = A @ B
    :return:
    """
    x, y = cuda.grid(2)
    ans = 0.

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    for i in range(B.shape[0]):
        ans += A[x, i] * B[i, y]

    C[x, y] = ans


def cu_matmul(A, B, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))
        B = cuda.to_device(B.astype(np.float32))

    assert len(A.shape) == 2 and len(B.shape) == 2

    res = cuda.device_array(shape=(A.shape[0], B.shape[1]), dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_matmul[grid_dim, block_dim](A, B, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_mattrans(A, B):
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = A[y, x]


# not tested
def cu_mattrans(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=(A.shape[1], A.shape[0]), dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_mattrans[grid_dim, block_dim](A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_matexp_2d(A, B):
    """
    :param A:
    :param B:
    :return: B = exp(A)
    """
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = math.exp(A[x, y])


def cu_matexp_2d(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_matexp_2d[grid_dim, block_dim](A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_matlog_2d(A, B):
    """
    :param A:
    :param B:
    :return: B = exp(A)
    """
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = math.log(A[x, y])


def cu_matlog_2d(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_matlog_2d[grid_dim, block_dim](A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_matabs_2d(A, B):
    """
    :param A:
    :param B:
    :return: B = abs(A)
    """
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = abs(A[x, y])


def cu_matabs_2d(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_matabs_2d[grid_dim, block_dim](A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_matpow_2d(A, B, n):
    """
    :param A:
    :param B:
    :return: B = exp(A)
    """
    x, y = cuda.grid(2)

    if x >= A.shape[0] or y >= A.shape[1]:
        return

    B[x, y] = math.pow(A[x, y], n)


def cu_matpow_2d(A, n, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_matpow_2d[grid_dim, block_dim](A, res, n)

    if to_host:
        return res.copy_to_host()
    else:
        return res


def cu_matsum_2d_1(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    I = cuda.to_device(np.ones(shape=(A.shape[1], 1), dtype=np.float32))
    res = cuda.device_array(shape=(A.shape[0], 1), dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_matmul[grid_dim, block_dim](A, I, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


def cu_matsum_2d_0(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    I = cuda.to_device(np.ones(shape=(1, A.shape[0]), dtype=np.float32))
    res = cuda.device_array(shape=(1, A.shape[1]), dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_matmul[grid_dim, block_dim](I, A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_matsum_1d(A, B):
    ans = 0.
    for i in range(A.shape[0]):
        ans += A[i]
    B[0] = ans


def cu_matsum_1d(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    res = cuda.device_array(shape=1, dtype=np.float32)
    _cu_matsum_1d[1, 1](A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_mat_broadcast_multiply_2d_0(A, B, C, a):
    x, y = cuda.grid(2)

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    C[x, y] = A[x, y] * math.pow(B[0, y], a)


def cu_mat_broadcast_multiply_2d_0(A, B, a=1, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))
        B = cuda.to_device(B.astype(np.float32))

    assert len(A.shape) == 2 and len(B.shape) == 2
    assert A.shape[1] == B.shape[1]
    assert B.shape[0] == 1

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_mat_broadcast_multiply_2d_0[grid_dim, block_dim](A, B, res, a)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_mat_broadcast_multiply_2d_1(A, B, C, a):
    x, y = cuda.grid(2)

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    C[x, y] = A[x, y] * math.pow(B[x, 0], a)


def cu_mat_broadcast_multiply_2d_1(A, B, a=1, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))
        B = cuda.to_device(B.astype(np.float32))

    assert len(A.shape) == 2 and len(B.shape) == 2
    assert A.shape[0] == B.shape[0]
    assert B.shape[1] == 1

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_mat_broadcast_multiply_2d_1[grid_dim, block_dim](A, B, res, a)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_mat_broadcast_add_2d_0(A, B, C, a):
    x, y = cuda.grid(2)

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    C[x, y] = A[x, y] + a * B[0, y]


def cu_mat_broadcast_add_2d_0(A, B, a=1, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))
        B = cuda.to_device(B.astype(np.float32))

    assert len(A.shape) == 2 and len(B.shape) == 2
    assert A.shape[1] == B.shape[1]
    assert B.shape[0] == 1

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_mat_broadcast_add_2d_0[grid_dim, block_dim](A, B, res, a)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_mat_broadcast_add_2d_1(A, B, C, a):
    x, y = cuda.grid(2)

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    C[x, y] = A[x, y] + a * B[x, 0]


def cu_mat_broadcast_add_2d_1(A, B, a=1, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))
        B = cuda.to_device(B.astype(np.float32))

    assert len(A.shape) == 2 and len(B.shape) == 2
    assert A.shape[0] == B.shape[0]
    assert B.shape[1] == 1

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_mat_broadcast_add_2d_1[grid_dim, block_dim](A, B, res, a)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_matmultply_2d(A, B, C, a):
    x, y = cuda.grid(2)

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    C[x, y] = A[x, y] * math.pow(B[x, y], a)


def cu_matmultply_2d(A, B, a=1, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))
        B = cuda.to_device(B.astype(np.float32))

    assert len(A.shape) == 2 and len(B.shape) == 2
    assert A.shape == B.shape

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_matmultply_2d[grid_dim, block_dim](A, B, res, a)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_matadd_2d(A, B, C, a):
    x, y = cuda.grid(2)

    if x >= C.shape[0] or y >= C.shape[1]:
        return

    C[x, y] = A[x, y] + a * B[x, y]


def cu_matadd_2d(A, B, a=1, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))
        B = cuda.to_device(B.astype(np.float32))

    assert len(A.shape) == 2 and len(B.shape) == 2
    assert A.shape == B.shape

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_matadd_2d[grid_dim, block_dim](A, B, res, a)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_mat_scalar_2d(A, B, a, b):
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = A[x, y] * a + b


def cu_mat_scalar_2d(A, a, b, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_mat_scalar_2d[grid_dim, block_dim](A, res, a, b)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_mat_max_2d_0(A, B, idx, from_host=False, to_host=False):
    x, y = cuda.grid(2)
    l_v = cuda.local.array(shape=1, dtype=float32)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    max_v = -math.inf
    max_idx = -1
    for i in range(A.shape[0]):
        l_v[0] = A[i, y]
        if l_v[0] > max_v:
            max_v = l_v[0]
            max_idx = i

    B[0, y] = max_v
    idx[0, y] = max_idx


def cu_mat_max_2d_0(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=(1, A.shape[1]), dtype=np.float32)
    idx = cuda.device_array(shape=(1, A.shape[1]), dtype=np.float32)

    grid_dim, block_dim = auto_detect(res.shape)
    _cu_mat_max_2d_0[grid_dim, block_dim](A, res, idx)

    if to_host:
        return res.copy_to_host(), idx.copy_to_host()
    else:
        return res, idx


@cuda.jit
def _cu_mat_max_2d_1(A, B, idx, from_host=False, to_host=False):
    x, y = cuda.grid(2)
    l_v = cuda.local.array(shape=1, dtype=float32)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    max_v = -math.inf
    max_idx = -1
    for i in range(A.shape[1]):
        l_v[0] = A[x, i]
        if l_v[0] > max_v:
            max_v = l_v[0]
            max_idx = i

    B[x, 0] = max_v
    idx[x, 0] = max_idx


@cuda.jit
def cu_mat_max_2d_1(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=(A.shape[0], 1), dtype=np.float32)
    idx = cuda.device_array(shape=(A.shape[0], 1), dtype=np.float32)

    grid_dim, block_dim = auto_detect(res.shape)
    _cu_mat_max_2d_0[grid_dim, block_dim](A, res, idx)

    if to_host:
        return res.copy_to_host(), idx.copy_to_host()
    else:
        return res, idx


@cuda.jit
def _cu_sigmoid_2d(A, B):
    """
    :param A:
    :param B:
    :return: B = sigmoid(A)
    """
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = 1 / (1 + math.exp(-A[x, y]))


def cu_sigmoid_2d(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_sigmoid_2d[grid_dim, block_dim](A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_sigmoid_2d_grad(A, B):
    """
    :param A:
    :param B:
    :return: B = sigmoid_grad(A)
    """
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = A[x, y] * (1 - A[x, y])


def cu_sigmoid_2d_grad(A=None, sigmoid_A=None, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    if sigmoid_A is None:
        sigmoid_A = cu_sigmoid_2d(A)

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_sigmoid_2d_grad[grid_dim, block_dim](sigmoid_A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_relu_2d(A, B):
    """
    :param A:
    :param B:
    :return: B = relu(A)
    """
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = max(0, A[x, y])
    pass


def cu_relu_2d(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_relu_2d[grid_dim, block_dim](A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_relu_2d_grad(A, B):
    """
    :param A:
    :param B:
    :return: B = relu_grad(A)
    """
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = (A[x, y] > 0)


def cu_relu_2d_grad(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_relu_2d_grad[grid_dim, block_dim](A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_tanh_2d(A, B):
    """
        :param A:
        :param B:
        :return: B = tanh(A)
        """
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = math.tanh(A[x, y])


def cu_tanh_2d(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_tanh_2d[grid_dim, block_dim](A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_tanh_2d_grad(A, B):
    """
    :param A:
    :param B:
    :return: B = tanh_grad(A)
    """
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = 1 - math.pow(A[x, y], 2)


def cu_tanh_2d_grad(A=None, tanh_A=None, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2

    if tanh_A is None:
        tanh_A = cu_tanh_2d(A)

    res = cuda.device_array(shape=A.shape, dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_tanh_2d_grad[grid_dim, block_dim](tanh_A, res)

    if to_host:
        return res.copy_to_host()
    else:
        return res


def detect_pooling_size(shape: tuple, kernel_size: int) -> object:
    assert len(shape) == 2

    output_size = (math.floor(shape / kernel_size) + 1)
    padding_size = output_size * kernel_size
    if padding_size != kernel_size:
        need_padding = True
    else:
        need_padding = False

    return output_size, padding_size, need_padding


@cuda.jit
def _cu_pool_padding(A, B, input_size, padding_size):
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    if x > input_size or y % padding_size > input_size:
        B[x, y] = 0
    else:
        B[x, y] = A[x, y]


def cu_pool_padding(A, dim_A: tuple, kernel_size, from_host=False, to_host=False):
    """
    :param A: flatterend 2d array
    :param dim_A: 4d dimension of A
    :param kernel_size:
    :param from_host:
    :param to_host:
    :return:
    """
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(dim_A) == 4
    assert isinstance(dim_A, tuple)
    assert dim_A[0] == dim_A[1]

    _, padding_size, need_padding = detect_pooling_size((dim_A[0], dim_A[1]), kernel_size)
    padding_shape = (padding_size, padding_size, dim_A[2], dim_A[3])
    if need_padding:
        res = cuda.device_array(shape=(padding_size, padding_size * dim_A[2] * dim_A[3]))
        grid_dim, block_dim = auto_detect(res.shape)
        _cu_pool_padding[grid_dim, block_dim](A, res, dim_A[0], padding_size)
    else:
        res = A

    if to_host:
        return res.copy_to_host(), padding_shape
    else:
        return res, padding_shape


@cuda.jit
def _cu_pool_depadding(A, B, input_size, padding_size):
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    B[x, y] = A[x, (y // input_size) * padding_size + (y % input_size)]


def cu_pool_depadding(A, dim_A: tuple, kernel_size, from_host=False, to_host=False):
    """
    :param A: flatterend 2d array, padded
    :param dim_A: 4d dimension of A, not padded
    :param kernel_size:
    :param from_host:
    :param to_host:
    :return:
    """
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(dim_A) == 4
    assert isinstance(dim_A, tuple)
    assert dim_A[0] == dim_A[1]

    _, padding_size, need_depadding = detect_pooling_size((dim_A[0], dim_A[1]), kernel_size)
    # padding_shape = (padding_size, padding_size, dim_A[2], dim_A[3])

    if need_depadding:
        res = cuda.device_array(shape=(dim_A[0], dim_A[1] * dim_A[2] * dim_A[3]))
        grid_dim, block_dim = auto_detect(res.shape)
        _cu_pool_depadding[grid_dim, block_dim](A, res, dim_A[0], padding_size)
    else:
        res = A

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_max_pool_2d(A, B, kernel_size):
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    max_v = -np.inf
    for i in range(kernel_size):
        for j in range(kernel_size):
            if A[x * kernel_size + i, y * kernel_size + j] > max_v:
                max_v = A[x * kernel_size + i, y * kernel_size + j]

    B[x, y] = max_v


def cu_max_pool_2d(A, kernel_size, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape == 2) and A.shape[1] % kernel_size == 0 and A[0] % kernel_size == 0

    res = cuda.device_array(shape=(int(A.shape[0] / kernel_size), int(A.shape[1] / kernel_size)), dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_max_pool_2d[grid_dim, block_dim](A, res, kernel_size)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_max_pool_2d_grad(A, B, kernel_size):
    x, y = cuda.grid(2)

    if x >= A.shape[0] or y >= A.shape[1]:
        return

    max_idx_x, max_idx_y = -1, -1
    max_v = -np.inf
    for i in range(kernel_size):
        for j in range(kernel_size):
            x_tmp = x * kernel_size + i
            y_tmp = y * kernel_size + j
            if A[x_tmp, y_tmp] > max_v:
                max_v = A[x_tmp, y_tmp]
                max_idx_x, max_idx_y = x_tmp, y_tmp

    B[max_idx_x, max_idx_y] = A[x, y]


def cu_max_pool_2d_grad(A, kernel_size, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape == 2) and A.shape[1] % kernel_size == 0 and A[0] % kernel_size == 0

    res = cuda.device_array(shape=(int(A.shape[0] * kernel_size), int(A.shape[1] * kernel_size)), dtype=np.float32)
    grid_dim, block_dim = auto_detect(A.shape)
    _cu_max_pool_2d_grad[grid_dim, block_dim](A, res, kernel_size)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_avg_pool_2d(A, B, kernel_size, pool_scale):
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    tmp_v = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            tmp_v += A[x * kernel_size + i, y * kernel_size + j]

    tmp_v /= pool_scale

    B[x, y] = tmp_v


def cu_avg_pool_2d(A, kernel_size, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape == 2) and A.shape[1] % kernel_size == 0 and A[0] % kernel_size == 0

    res = cuda.device_array(shape=(int(A.shape[0] / kernel_size), int(A.shape[1] / kernel_size)), dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    pool_scale = kernel_size * kernel_size
    _cu_avg_pool_2d[grid_dim, block_dim](A, res, kernel_size, pool_scale)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_avg_pool_2d_grad(A, B, kernel_size, pool_scale):
    x, y = cuda.grid(2)

    if x >= A.shape[0] or y >= A.shape[1]:
        return

    for i in range(kernel_size):
        for j in range(kernel_size):
            B[x * kernel_size + i, y * kernel_size + j] = A[x, y] / pool_scale


def cu_avg_pool_2d_grad(A, kernel_size, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape == 2) and A.shape[1] % kernel_size == 0 and A[0] % kernel_size == 0

    res = cuda.device_array(shape=(int(A.shape[0] * kernel_size), int(A.shape[1] * kernel_size)), dtype=np.float32)
    grid_dim, block_dim = auto_detect(A.shape)
    pool_scale = kernel_size * kernel_size
    _cu_avg_pool_2d_grad[grid_dim, block_dim](A, res, kernel_size, pool_scale)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_mat_4d_to_2d(A, B):
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    i = x
    j = y % A.shape[1]
    c = (y % (A.shape[1] * A.shape[2])) // A.shape[1]
    k = y // (A.shape[1] * A.shape[2])

    B[x, y] = A[i, j, c, k]


def cu_mat_4d_to_2d(A, from_host=False, to_host=False):
    """

    :param A:
    :param from_host:
    :param to_host:
    :return: A:4d matrix, shape: A's shape
    """
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 4 and A.shape[0] == A.shape[1]

    res = cuda.device_array(shape=(A.shape[0], A.shape[1] * A.shape[2] * A.shape[3]), dtype=np.float32)
    grid_dim, block_dim = auto_detect(res.shape)
    _cu_mat_4d_to_2d[grid_dim, block_dim](A, res)

    if to_host:
        return res.copy_to_host(), A.shape
    else:
        return res, A.shape


@cuda.jit
def _cu_mat_2d_to_4d(A, B):
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    i = x
    j = y % B.shape[1]
    c = (y % (B.shape[1] * B.shape[2])) // B.shape[1]
    k = y // (B.shape[1] * B.shape[2])

    B[i, j, c, k] = A[x, y]


def cu_mat_2d_to_4d(A, dim, from_host=False, to_host=False):
    """
    :param A:
    :param from_host:
    :param to_host:
    :return: A:2d matrix, shape: A's shape
    """
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert len(A.shape) == 2 and A.shape[0] == dim[0]

    res = cuda.device_array(shape=(dim[0], dim[1], dim[2], dim[3]), dtype=np.float32)
    grid_dim, block_dim = auto_detect(A.shape)
    _cu_mat_2d_to_4d[grid_dim, block_dim](A, res)

    if to_host:
        return res.copy_to_host(), A.shape
    else:
        return res, A.shape


def cu_mat_flattern(A, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    dim_x = 1
    for i in range(len(A.shape) - 1):
        dim_x * A[i]

    res = A.reshape(dim_x, A.shape[3])

    if to_host:
        return res.copy_to_host(), A.shape
    else:
        return res, A.shape


def cu_mat_deflattern(A, shape, from_host=False, to_host=False):
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    assert isinstance(shape, tuple)
    # might not be necessary
    assert len(A.shape) == 2 and len(shape) == 4

    res = A.reshape(shape)

    if to_host:
        return res.copy_to_host()
    else:
        return res


@cuda.jit
def _cu_conv_im2col(A, B, output_span, output_sz, kernel_sz, in_c):
    x, y = cuda.grid(2)

    if x >= B.shape[0] or y >= B.shape[1]:
        return

    batch_idx = x // (output_span)
    in_c_idx = y % in_c
    kernel_idx = y // in_c
    kernel_idx_y = kernel_idx % kernel_sz
    kernel_idx_x = kernel_idx // kernel_sz
    output_idx = x % (output_span)
    output_idx_y = output_idx % output_sz
    output_idx_x = output_idx // output_sz

    B[x, y] = A[batch_idx, output_idx_x + kernel_idx_x, output_idx_y + kernel_idx_y, in_c_idx]


def cu_conv_im2col(A, kernel_shape, from_host=False, to_host=False):
    """
    :param inpt: shape=(m,m,C1,k)
    :return: shape=(c1*a*a,(n+2p-a+1)^2*k)
    """
    if from_host:
        A = cuda.to_device(A.astype(np.float32))

    batch_sz, h, w, in_c = A.shape
    kernel_sz = kernel_shape[0]
    output_sz = (h - kernel_sz) + 1

    res = cuda.device_array(shape=(batch_sz * output_sz * output_sz, kernel_sz * kernel_sz * in_c)
                            , dtype=np.float32)
    output_span = output_sz * output_sz

    grid_dim, block_dim = auto_detect(res.shape)
    _cu_conv_im2col[grid_dim, block_dim](A, res, output_span, output_sz, kernel_sz, in_c)

    if to_host:
        return res.copy_to_host()
    else:
        return res


def cu_max_pool_4d(inpt, from_host=False, to_host=False):
    """
    :param inpt: shape=(m,m,C1,k)
    :return: shape=(c1*a*a,(n+2p-a+1)^2*k)
    """
    raise NotImplementedError


def cu_avg_pool_4d(inpt, from_host=False, to_host=False):
    """
    :param inpt: shape=(m,m,C1,k)
    :return: shape=(c1*a*a,(n+2p-a+1)^2*k)
    """
    raise NotImplementedError


def cu_conv_reshape_kernel(K, from_host=False, to_host=False):
    """
    :param K: shape=(C1,C2,a,a)
    :return: shape=(C2,a*a*C1)
    """
    raise NotImplementedError


def cu_conv_dereshape_kernel(K, from_host=False, to_host=False):
    """
    :param K: shape=(C2,a*a*C1)
    :return: shape=(C1,C2,a,a)
    """
    raise NotImplementedError


def cu_conv_reshape_bias(K, from_host=False, to_host=False):
    """
    :param K: shape=(C1,C2,a,a)
    :return: shape=(C2,a*a*C1)
    """
    raise NotImplementedError


def cu_conv_dereshape_bias(K, from_host=False, to_host=False):
    """
    :param K: shape=(C2,a*a*C1)
    :return: shape=(C1,C2,a,a)
    """
    raise NotImplementedError


def cu_conv_col2im(inpt, from_host=False, to_host=False):
    """
    :param inpt: shape=(c1*a*a,(n+2p-a+1)^2*k)
    :return: shape=(m,m,C1,k)
    """
    raise NotImplementedError


def cu_conv_reshape_output(output, from_host=False, to_host=False):
    """
    :var m = (n+2p-a+1)
    :param output: shape=(C2,m*m*k)
    :return: shape=(m,m,C2,k)
    """
    raise NotImplementedError


def cu_conv_dereshape_output(output, from_host=False, to_host=False):
    """
    :var m = (n+2p-a+1)
    :param output: shape=(m,m,C2,k)
    :return: shape=(C2,m*m*k)
    """
    raise NotImplementedError

# def _cu (A, B, output_span, output_sz, kernel_sz, in_c):
#     x, y = cuda.grid(2)
#
#     if x >= B.shape[0] or y >= B.shape[1]:
#         return
#
#     batch_idx = x // (output_span)
#     in_c_idx = y % in_c
#     kernel_idx = y // in_c
#     kernel_idx_y = kernel_idx % kernel_sz
#     kernel_idx_x = kernel_idx // kernel_sz
#     output_idx = x % (output_span)
#     output_idx_y = output_idx % output_sz
#     output_idx_x = output_idx // output_sz
#
#     B[x, y] = A[batch_idx, output_idx_x + kernel_idx_x, output_idx_y + kernel_idx_y, in_c_idx]


# def cu_conv_im2col(A, kernel_shape, from_host=False, to_host=False):
#     """
#     :param inpt: shape=(m,m,C1,k)
#     :return: shape=(c1*a*a,(n+2p-a+1)^2*k)
#     """
#     if from_host:
#         A = cuda.to_device(A.astype(np.float32))
#
#     batch_sz, h, w, in_c = A.shape
#     kernel_sz = kernel_shape[0]
#     output_sz = (h - kernel_sz) + 1
#
#     res = cuda.device_array(shape=(batch_sz * output_sz * output_sz, kernel_sz * kernel_sz * in_c)
#                             , dtype=np.float32)
#     output_span = output_sz * output_sz
#
#     grid_dim, block_dim = auto_detect(res.shape)
#     _cu_conv_im2col[grid_dim, block_dim](A, res, output_span, output_sz, kernel_sz, in_c)
#
#     if to_host:
#         return res.copy_to_host()
#     else:
#         return res
