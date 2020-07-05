import numpy as np


def im2col(x, kernel_shape):
    """Transform padded image into column matrix.
    :param x: padded inputs of shape (B, in_h, in_w, in_c)
    :return col: column matrix of shape (B*out_h*out_w, k_h*k_h*inc)
    """
    batch_sz, h, w, in_c = x.shape
    k_h, k_w, = kernel_shape[0], kernel_shape[1]
    out_h = (h - k_h) + 1
    out_w = (w - k_w) + 1
    col = np.empty(shape=(batch_sz * out_h * out_w, k_h * k_w * in_c))
    batch_span = out_w * out_h
    for r in range(out_h):
        r_start = r
        matrix_r = r * out_w
        for c in range(out_w):
            c_start = c
            patch = x[:, r_start: r_start + k_h, c_start: c_start + k_w, :]
            patch = patch.reshape(batch_sz, -1)
            col[matrix_r + c::batch_span, :] = patch
    return col


def col2im(x, kernel_shape):
    raise NotImplementedError
