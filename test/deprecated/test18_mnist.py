from models import Linear1
from sklearn import datasets
from core.Optimizers import sgd, bgd
from core.Functions import one_hot_f
import numpy as np
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# origin = np.random.randn(3, 28, 28).astype(np.float32)
# input_image = train_images[0, :, :].reshape(1,28,28)
train_images = train_images / 255
test_images = test_images / 255
train_labels = one_hot_f(train_labels, num_classes=10)
test_labels = one_hot_f(test_labels, num_classes=10)

Linear1.compile()
optimizer = bgd(0.01)
optimizer.fit(Linear1, train_images, train_labels, epoch=30, batch_size=16)
# print(Linear1.val(np.transpose(input_image)))
