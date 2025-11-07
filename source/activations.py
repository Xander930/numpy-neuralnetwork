import numpy as np


def sigmoid(x):
    """
    sigmoid activation function: inputs np.ndarray and returns np.ndarray
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    """
    derivative of the sigmoid function, input ndarray and returns np.ndarray
    """
    s = sigmoid(x)
    return s * (1 - s)
