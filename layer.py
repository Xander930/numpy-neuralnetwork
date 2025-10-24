import numpy as np


inputs = []
weights = []
bias = []

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))
    def forward(self):
        self.output = np.dot(inputs, np.array(self.weights).T) + self.bias

