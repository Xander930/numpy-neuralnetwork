from .activations import sigmoid, sigmoid_prime
from .utils import Layer, mse_prime


class NeuralNet:
    def __init__(self, size_in, size_hidden, size_out, learning_rate):
        self.learning_rate = learning_rate
        self.layer1 = Layer(size_in, size_hidden)
        self.layer2 = Layer(size_hidden, size_out)
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def forward(self, X):
        self.z1 = self.layer1.forward(X)
        self.a1 = sigmoid(self.z1)
        self.z2 = self.layer2.forward(self.a1)
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        dL_da2 = mse_prime(y, self.a2)
        da2_dz2 = sigmoid_prime(self.z2)
        dz1 = dL_da2 * da2_dz2
