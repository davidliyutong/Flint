"""
This script tests the performance of our tools
"""
from numba import cuda
from tensorflow import keras
from core.Layers.conv import conv2d
from core.Layers.pool import max_pool
from core.Layers import relu, softmax
from core.Layers import dense
from core.Layers import flatten
import time

conv1 = conv2d((1, 28, 28), output_channel=32, kernel_size=3)
activ1 = relu((3, 26, 26))
pool1 = max_pool(input_dim=(32, 26, 26), kernel_sz=2)
conv2 = conv2d((32, 13, 13), output_channel=32, kernel_size=3)
activ2 = relu((32, 11, 11))
pool2 = max_pool(input_dim=(32, 11, 11), kernel_sz=3)
activ3 = relu((32, 4, 4))
flat = flatten((32, 4, 4))
linear_layer = dense(32 * 4 * 4, 10)
softmax_layer = softmax(10)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# origin = np.random.randn(3, 28, 28).astype(np.float32)
input_image = train_images[0, :, :].reshape(1, 28, 28)
input_image_gpu = cuda.to_device(input_image)

cnt = 0
t_start = time.time()

while True:
    cnt += 1
    input_image_gpu = cuda.to_device(input_image)
    out1_conv = conv1.forward(input_image_gpu)
    out1_activ = activ1.forward(out1_conv)
    out1_pool = pool1.forward(out1_activ)

    out2_conv = conv2.forward(out1_pool)
    out2_activ = activ2.forward(out2_conv)
    out2_pool = pool2.forward(out2_activ)

    out3_activ = activ3.forward(out2_pool)
    out3_flat = flat.forward(out3_activ)
    out3_linear = linear_layer.forward(out3_flat)
    out3_softmax = softmax_layer.forward(out3_linear)
    if time.time() - t_start > 1:
        print('[ fps ]', cnt / (time.time() - t_start))
        cnt = 0
        t_start = time.time()
