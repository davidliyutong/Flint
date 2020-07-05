import numpy as np
import numba


def im2col(x, input_dim, output_dim, kernel_dim, batch_size):
    """Transform padded image into column matrix.
    :reference: https://github.com/borgwang/tinynn
    """
    assert len(input_dim) == 3
    assert x.shape[0:3] == input_dim
    assert x.shape[0] == x.shape[1]

    input_size = input_dim[0]
    input_channel = input_dim[2]
    kernel_size = kernel_dim[0]
    output_size = output_dim[0]
    output_channel = output_dim[2]

    col = np.zeros(shape=(input_channel * kernel_size * kernel_size,
                          output_size * output_size * batch_size),
                   dtype=np.float32)

    span_x = kernel_size * kernel_size
    span_y = output_size * output_size

    for i in range(input_channel):
        for j in range(batch_size):
            for m in range(output_size):
                for n in range(output_size):
                    try:
                        col[i * span_x:(i + 1) * span_x, j * span_y + m * output_size + n] = x[m:m + kernel_size,
                                                                                             n:n + kernel_size, i,
                                                                                             j].reshape(-1)
                    except:
                        print('e')
    return col


def col2im(x, input_dim, output_dim, kernel_dim, batch_size):
    assert len(x.shape) == 2

    input_size = input_dim[0]
    input_channel = input_dim[2]
    kernel_size = kernel_dim[0]
    output_size = output_dim[0]
    output_channel = output_dim[2]

    assert x.shape[0] == input_channel * kernel_size * kernel_size
    assert x.shape[1] == output_size * output_size * batch_size

    span_x = kernel_size * kernel_size
    span_y = output_size * output_size

    im = np.zeros(shape=(input_size, input_size, input_channel, batch_size),
                  dtype=np.float32)

    span_x = kernel_size * kernel_size
    span_y = output_size * output_size

    for i in range(input_channel):
        for j in range(batch_size):
            for m in range(output_size):
                for n in range(output_size):
                    try:
                        im[m:m + kernel_size, n:n + kernel_size, i, j] = x[i * span_x:(i + 1) * span_x,
                                                                         j * span_y + m * output_size + n].reshape(
                            kernel_size, kernel_size)
                    except:
                        print('e')
    return im


def reshape_output(x, output_dim, batch_size):
    assert len(x.shape) == 2

    output_size = output_dim[0]
    output_channel = output_dim[2]

    span_y = output_size * output_size
    output = np.zeros(shape=(*output_dim, batch_size), dtype=np.float32)

    for i in range(batch_size):
        for j in range(output_channel):
            output[:, :, j, i] = x[j, i * span_y:(i + 1) * span_y].reshape(output_size, output_size)

    return output


def dereshape_output(x, output_dim, batch_size):
    assert len(x.shape) == 4

    output_size = output_dim[0]
    output_channel = output_dim[2]

    span_y = output_size * output_size
    output = np.zeros(shape=(output_channel, output_size * output_size * batch_size),
                      dtype=np.float32)

    for i in range(batch_size):
        for j in range(output_channel):
            output[j, i * span_y:(i + 1) * span_y] = x[:, :, j, i].reshape(-1)

    return output


def reshape_kernel(x, kernel_dim):
    assert len(kernel_dim) == 4

    kernel_size = kernel_dim[0]
    input_channel = kernel_dim[2]
    output_channel = kernel_dim[3]

    output = np.zeros(shape=(output_channel, input_channel * kernel_size * kernel_size),
                      dtype=np.float32)

    span_y = kernel_size * kernel_size
    for i in range(output_channel):
        for j in range(input_channel):
            output[i, j * span_y:(j + 1) * span_y] = x[:, :, j, i].reshape(-1)

    return output


def dereshape_kernel(x, kernel_dim):
    assert len(kernel_dim) == 4

    kernel_size = kernel_dim[0]
    input_channel = kernel_dim[2]
    output_channel = kernel_dim[3]

    output = np.zeros(shape=(kernel_size, kernel_size, input_channel, output_channel),
                      dtype=np.float32)

    span_y = kernel_size * kernel_size
    for i in range(output_channel):
        for j in range(input_channel):
            output[:, :, j, i] = x[i, j * span_y:(j + 1) * span_y].reshape(kernel_size, kernel_size)

    return output


def reshape_bias(x, output_dim):
    assert len(x.shape) == 3

    output_size = output_dim[0]
    output_channel = output_dim[2]

    output = np.zeros(shape=(output_channel, output_size * output_size),
                      dtype=np.float32)

    span_y = output_size * output_size

    for i in range(output_channel):
        output[i, :] = x[:, :, i].reshape(-1)

    return output


def dereshape_bias(x, output_dim):
    assert len(x.shape) == 2

    output_size = output_dim[0]
    output_channel = output_dim[2]

    output = np.zeros(shape=(output_size, output_size, output_channel),
                      dtype=np.float32)

    span_y = output_size * output_size

    for i in range(output_channel):
        output[:, :, i] = x[i, :].reshape(output_size, output_size)

    return output


def broadcast_bias(x, batch_size):
    assert len(x.shape) == 2

    output = np.zeros(shape=(x.shape[0], x.shape[1] * batch_size))

    for i in range(batch_size):
        output[:, i * x.shape[1]: (i + 1) * x.shape[1]] = x

    return output


def debroadcast_bias(x, batch_size):
    assert len(x.shape) == 2
    assert x.shape[1] % batch_size == 0

    output = np.zeros(shape=(x.shape[0], x.shape[1] // batch_size))
    span_y = x.shape[1] // batch_size

    for i in range(batch_size):
        output += x[:, i * span_y: (i + 1) * span_y]

    output /= batch_size

    return output


if __name__ == "__main__":
    # kernel: (a,a,c1,c2)

    input_dim = (28, 28, 32)
    output_dim = (26, 26, 16)
    kernel_dim = (3, 3, 32, 16)
    batch_size = 1
    A = np.random.randn(*input_dim, batch_size)
    K = np.random.randn(*kernel_dim)
    B = np.random.randn(26, 26, 16)

    A_col = im2col(A, input_dim, output_dim, kernel_dim, batch_size)
    K_reshaped = reshape_kernel(K, kernel_dim)
    B_reshaped = reshape_bias(B, output_dim)
    O_with_out_bias = np.matmul(K_reshaped, A_col)
    B_reshaped_broadcasted = broadcast_bias(B_reshaped, batch_size)
    O_with_bias = O_with_out_bias + B_reshaped_broadcasted
    O_reshaped = reshape_output(O_with_bias, output_dim, batch_size)

    grad = np.random.randn(26, 26, 16, 1)
    grad_dereshaped = dereshape_output(grad, output_dim, batch_size)

    B_grad_reshaped = debroadcast_bias(grad_dereshaped, batch_size)
    B_grad = dereshape_bias(B_grad_reshaped, output_dim)
    K_grad_reshaped = np.matmul(grad_dereshaped, np.transpose(A_col))
    K_grad = dereshape_kernel(K_grad_reshaped, kernel_dim)

    A_grad_col = np.matmul(np.transpose(K_reshaped), grad_dereshaped)
    A_grad = col2im(A_grad_col, input_dim, output_dim, kernel_dim, batch_size)

    B -= B_grad * 0.01
    K -= K_grad * 0.01
    A -= A_grad * 0.01

    print('[ success ]')
