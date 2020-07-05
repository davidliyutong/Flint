from models import CNN1, CNN2, CNN3
from sklearn import datasets
from core.Optimizers import sgd, bgd
from core.Functions import one_hot_f
import numpy as np
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# origin = np.random.randn(3, 28, 28).astype(np.float32)
# input_image = train_images[0, :, :].reshape(1,28,28)
train_images = np.expand_dims(train_images / 255, axis=1)
test_images = np.expand_dims(test_images / 255, axis=1)
train_labels = one_hot_f(train_labels, num_classes=10)
test_labels = one_hot_f(test_labels, num_classes=10)

CNN2.compile()
optimizer = sgd(0.01)
optimizer.fit(CNN2, train_images, train_labels, epoch=30, batch_size=1)
# optimizer = bgd(0.01)
# optimizer.fit(CNN2, train_images, train_labels, epoch=30, batch_size=16)

CNN3.compile()
optimizer = sgd(0.01)
optimizer.fit(CNN3, train_images, train_labels, epoch=30, batch_size=1)
