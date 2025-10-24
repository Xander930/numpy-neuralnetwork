import numpy as np

inputs = []
weights = []
bias = []

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        output = np.dot(inputs, np.array(self.weights).T) + self.bias
        return output
