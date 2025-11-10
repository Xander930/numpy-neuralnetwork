import numpy as np


class Layer:
    """
    General layer class.

    3 variables in cache:
    - self.weights, weight matrix for layer cached during __init__
    - self.bias, bias matrix for layer cached during __init__

    -self.output, layer output matrix cached during forward pass

    __init__ method: specifcy input matrix size and number of neurons in layer

    forward method: pass inputs through layer.
    """

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias
        return self.output


def loss_mse(y, y_pred):
    """
    mean squared error loss function, inputs 2x np,ndarray and returns float.
    """
    return np.mean((y - y_pred) ** 2)


def mse_prime(y, y_pred):
    """
    derivative of MSE function, inputs 2x np.ndarray and returns float.
    """
    return 2 * (y_pred - y)
