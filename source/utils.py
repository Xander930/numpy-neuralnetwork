import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, np.array(self.weights)) + self.bias
        return self.output


def loss_mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def mse_prime(y, y_pred):
    return 2 * (y_pred - y) / y.size
