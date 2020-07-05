import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class custom_sgd_optimizer(object):
    def __init__(self, lr=1e-3):
        self.lr = lr

    def apply_gradients(self, gradient, model_variables):
        for var in zip(gradient, model_variables):
            var[1].assign_sub(var[0] * self.lr)


class custom_adam_optimizer(object):
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = eps
        self.lr = lr
        self.is_ready = False
        self._m = None
        self._v = None
        self._t = 0
        self._num_of_grads = 0

    def _initiate_parameters(self, num_of_grads: int):
        self._num_of_grads = num_of_grads
        self._m = []
        self._v = []
        v = tf.Variable(0.)

        for i in range(self._num_of_grads):
            self._m.append(v)
            self._v.append(v)

        self.is_ready = True


    def apply_gradients(self, gradient, model_variables):
        self._t += 1

        if not self.is_ready:
            self._initiate_parameters(len(gradient))

        grad_step = []

        for i in range(self._num_of_grads):
            self._m[i] = self._beta1 * self._m[i] + (1 - self._beta1) * (gradient[i])
            self._v[i] = self._beta2 * self._v[i] + (1 - self._beta2) * (gradient[i] ** 2)
            grad_step.append((self._m[i] / (1 - self._beta1 ** self._t)) / ((self._v[i] / (1 - self._beta2 ** self._t)) ** 0.5 + self._epsilon))

        for var in zip(grad_step, model_variables):
            var[1].assign_sub(var[0] * self.lr)



def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(gradients, model.trainable_variables)

    train_loss(loss)
    train_accuracy(labels, predictions)


def test_step(model, images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


if __name__ == '__main__':
    EPOCHS = 5
    model = MyModel()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = custom_adam_optimizer(lr=1e-3)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(model, images, labels, optimizer)

        for test_images, test_labels in test_ds:
            test_step(model, test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
