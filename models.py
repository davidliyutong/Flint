from core.Layers.conv import conv2d
from core.Layers.pool import max_pool
from core.Layers import relu, dense, flatten, tanh, sigmoid
from core.Modules import sequential
from core.Functions.loss import softmax_cross_entropy

CNN1 = sequential(conv2d(input_shape=(28, 28, 1), output_ch=32, kernel_sz=3),
                  relu(),
                  max_pool(kernel_sz=2),
                  conv2d(output_ch=32, kernel_sz=3),
                  relu(),
                  max_pool(kernel_sz=2),
                  flatten(),
                  dense(output_shape=10),
                  loss_f=softmax_cross_entropy()
                  )

CNN2 = sequential(conv2d(input_shape=(28, 28, 1), output_ch=6, kernel_sz=5),
                  relu(),
                  max_pool(kernel_sz=2),
                  conv2d(output_ch=16, kernel_sz=5),
                  relu(),
                  max_pool(kernel_sz=2),
                  flatten(),
                  dense(output_shape=10),
                  loss_f=softmax_cross_entropy()
                  )

CNN3 = sequential(conv2d(input_shape=(28, 28, 1), output_ch=1, kernel_sz=2),
                  relu(),
                  max_pool(kernel_sz=2),
                  flatten(),
                  dense(output_shape=10),
                  loss_f=softmax_cross_entropy()
                  )

# LeNet-5
CNN4 = sequential(conv2d(input_shape=(28, 28, 1), output_ch=6, kernel_sz=5),
                  max_pool(kernel_sz=2),
                  conv2d(output_ch=16, kernel_sz=5),
                  max_pool(kernel_sz=2),
                  flatten(),
                  dense(output_shape=120),
                  relu(),
                  dense(output_shape=84),
                  dense(output_shape=10),
                  loss_f=softmax_cross_entropy()
                  )

# brought from andy
CNN5 = sequential(conv2d(input_shape=(28, 28, 1), output_ch=16, kernel_sz=3),
                  relu(),
                  max_pool(kernel_sz=2),
                  flatten(),
                  dense(output_shape=64),
                  relu(),
                  dense(output_shape=10),
                  loss_f=softmax_cross_entropy()
                  )

Linear1 = sequential(flatten(input_shape=(28, 28, 1)),
                     dense(output_shape=64),
                     relu(),
                     dense(output_shape=16),
                     relu(),
                     dense(output_shape=10),
                     loss_f=softmax_cross_entropy()
                     )

Linear2 = sequential(dense(input_shape=4, output_shape=64),
                     tanh(),
                     dense(output_shape=32),
                     sigmoid(),
                     dense(output_shape=3),
                     loss_f=softmax_cross_entropy()
                     )
# from tensorflow's tutorial
Linear3 = sequential(flatten(input_shape=(28, 28, 1)),
                     dense(output_shape=200),
                     relu(),
                     dense(output_shape=128),
                     relu(),
                     dense(output_shape=10),
                     loss_f=softmax_cross_entropy()
                     )
