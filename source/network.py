import numpy as np

from .activations import sigmoid, sigmoid_prime
from .utils import Layer, loss_mse, mse_prime


class NeuralNet:
    """
    Neural network class.

    Initialize with input, hidden, and output sizes as well as learning rate.

    Upon init, stores layers, layer and activation outputs, and gradients as pre-allocated variables.
    """

    def __init__(self, size_in, size_hidden, size_out, learning_rate):
        # pre-allocate variable for learning rate
        self.learning_rate = learning_rate

        # initialize layers
        self.layer1 = Layer(size_in, size_hidden)
        self.layer2 = Layer(size_hidden, size_out)

        # pre-allocate variables for layer/activation output
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

        # pre-allocate variables for layer/activation gradients
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

    def forward(self, X):
        """
        Forward pass function. Inputs np.ndarray (X) and returns np.ndarray.

        Passes input X through layers and activation functions in series, layer 1 -> activation -> layer 2 -> activation.
        """
        # forward pass layer 1 -> activation
        self.z1 = self.layer1.forward(X)
        self.a1 = sigmoid(self.z1)

        # forward pass layer 2 -> activation
        self.z2 = self.layer2.forward(self.a1)
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        """
        Backward pass function. Inputs np.ndarray (X) and np.ndarray (y).

        Computes gradients and assigns them to the variables pre-allocated during initialization.

        Flows in reverse order from forward(), activation -> layer 2 -> activation -> layer 1.
        """
        # store batch size for gradient averaging
        m = X.shape[0]

        # compute derivative of MSE with respect to output activations
        dL_da2 = mse_prime(y, self.a2)

        # compute derivative of activation function at layer 2
        da2_dz2 = sigmoid_prime(self.z2)

        # chain rule to compute gradient with respect to z2
        dz2 = dL_da2 * da2_dz2

        # compute weight gradient for layer 2
        self.dW2 = np.dot(self.a1.T, dz2) / m

        # compute bias gradient for layer 2
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # transpose weights from layer 2
        dz2_da1 = self.layer2.weights.T

        # chain rule to compute gradient of layer 2 weights with respect to layer 1 activations
        da1 = np.dot(dz2, dz2_da1)

        # compute derivative of activation function at layer 1
        da1_dz1 = sigmoid_prime(self.z1)

        # chain rule to compute gradient with respect to z1
        dz1 = da1 * da1_dz1

        # compute weight gradient for layer 1
        self.dW1 = np.dot(X.T, dz1) / m

        # compute bias gradient for layer 1
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m

    def update_weights(self):
        """
        Updates weights and bias matrices for neuron layers using stored gradients computed during backpropagation.

        Works in the same direction as backward().
        """
        # update weights and biases for layer 2 (scaled by learning rate)
        self.layer2.weights -= self.learning_rate * self.dW2
        self.layer2.bias -= self.learning_rate * self.db2

        # update weights and biases for layer 1 (scaled by learning rate)
        self.layer1.weights -= self.learning_rate * self.dW1
        self.layer1.bias -= self.learning_rate * self.db1

    def train(self, X, y, epochs, verbose=True):
        """
        Trains the neural network. Inputs np.ndarray (X), np,ndarray (y), int (epochs), and Bool (verbose). Returns list.

        Runs previously specified methods in sequence for n iterations, where n is the number of epochs.
        Training occurs using the following flow: forward pass -> loss calculation -> backpropagation -> weight and bias update.
        Function appends the calculated loss every epoch to a list for training analysis.

        if verbose is set to True, will print epoch number and loss every 1000 epochs as well as the pentultimate iteration.
        """
        # allocate loss history list
        loss_hist = []

        # training iteration loop
        for epoch in range(epochs):

            # activate forward pass method
            preds = self.forward(X)

            # calculate loss after forward pass
            loss = loss_mse(y, preds)

            # append iteration loss to history list
            loss_hist.append(loss)

            # backpropagate
            self.backward(X, y)

            # update weights and biases
            self.update_weights()

            # print epoch + loss if verbose
            if verbose and epoch % 1000 == 0 or epoch == epochs - 1:
                print(f"epoch {epoch:5d} | loss: {loss:.6f}")
        return loss_hist

    def predict(self, X, threshold):
        """
        Make predictions based on inputs using network. Inputs np.ndarray (X) and float (threshold). Returns binary prediction as int.

        Performs forward pass method in order to predict outputs. Outputs are then clipped to either 0 or 1 using the provided threshold.
        """
        # activate forward pass method
        preds = self.forward(X)

        # return predictions clipped to binary int using threshold
        return (preds >= threshold).astype(int)
