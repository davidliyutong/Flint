"""
Record the time to compile each layer
"""

from numba import cuda
from tensorflow import keras
from core.Layers.conv import conv2d
from core.Layers.pool import max_pool
from core.Layers import relu, softmax
from test.deprecated import one_hot
from dense import linear
import time

conv1 = conv2d((1, 28, 28), output_channel=3, kernel_size=3)
pool1 = max_pool(input_dim=(3, 26, 26), kernel_sz=2)
activ1 = relu((3, 13, 13))
conv2 = conv2d((3, 13, 13), output_channel=3, kernel_size=3)
pool2 = max_pool(input_dim=(3, 11, 11), kernel_sz=3)
activ2 = relu((3, 4, 4))
linear_layer = linear(3 * 4 * 4, 10)
softmax_layer = softmax(10)
one_hot_layer = one_hot(10)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# origin = np.random.randn(3, 28, 28).astype(np.float32)
input_image = train_images[0, :, :].reshape(1, 28, 28)
input_image_gpu = cuda.to_device(input_image)

t_start = time.time()
out1_conv = conv1.forward(input_image_gpu)
print('[ time:conv ]', time.time() - t_start)
t_start = time.time()
out1_pool = pool1.forward(out1_conv)
print('[ time:pool ]', time.time() - t_start)
t_start = time.time()
out1_activ = activ1.forward(out1_pool)
print('[ time:activ ]', time.time() - t_start)
t_start = time.time()

out2_conv = conv2.forward(out1_activ)
out2_pool = pool2.forward(out2_conv)
out2_activ = activ2.forward(out2_pool)
print('[ time:compound ]', time.time() - t_start)
t_start = time.time()

out3_linear = linear_layer.forward(out2_activ)
print('[ time:dense ]', time.time() - t_start)
t_start = time.time()
out3_softmax = softmax_layer.forward(out3_linear)
print('[ time:softmax ]', time.time() - t_start)
t_start = time.time()
out3 = one_hot_layer.forward(out3_softmax)
print('[ time:on_hot ]', time.time() - t_start)
