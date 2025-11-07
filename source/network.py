import numpy as np
from .activations import sigmoid, sigmoid_prime
from .utils import Layer, loss_mse, mse_prime


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
        dz2 = dL_da2 * da2_dz2
        self.dW2 = np.dot(self.a1.T, dz2) / m
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz2_da1 = self.layer2.weights.T
        da1 = np.dot(dz2, dz2_da1)
        da1_dz1 = sigmoid_prime(self.z1)
        dz1 = da1 * da1_dz1
        self.dW1 = np.dot(X.T, dz1) / m
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m

    def update_weights(self):
        self.layer2.weights -= self.learning_rate * self.dW2
        self.layer2.bias -= self.learning_rate * self.db2
        self.layer1.weights -= self.learning_rate * self.dW1
        self.layer1.bias -= self.learning_rate * self.db1

    def train(self, X, y, epochs, verbose=True):
        loss_hist = []
        for epoch in range(epochs):
            preds = self.forward(X)
            loss = loss_mse(y, preds)
            loss_hist.append(loss)
            self.backward(X, y)
            self.update_weights()
        if verbose and (epoch % 1000 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:5d} | loss: {loss:.6f}")
        return loss_hist

    def predict(self, X, threshold):
        preds = self.forward(X)
        return (preds >= threshold).astype(int)
