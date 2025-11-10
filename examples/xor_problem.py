import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from source.network import NeuralNet


def main():
    """
    Main loop for actually running and training the network
    """

    #input (X) and output (y) variable matrices
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    #print training data matrices to terminal
    print("\nTraining Data")
    for i in range(len(X)):
        print(f"    Input: {X[i]} -> Output: {y[i][0]}")

    #begin training
    print("Beginning Training...")

    #initialize neural network with parameters
    nn = NeuralNet(size_in=2, size_hidden=4, size_out=1, learning_rate=0.5)

    #train neural network for a specified number of epochs
    loss_hist = nn.train(X, y, epochs=50000, verbose=True)

    print("Results: ")

    #forward pass through newly trained network
    predictions = nn.forward(X)

    #generate output predictions in binary form
    binary_preds = nn.predict(X, threshold=0.5)

    #for-loop to print predistions
    print("\nPredictions: ")
    for i in range(len(X)):
        print(
            f"  Input: {X[i]}  ->  Predicted: {predictions[i][0]:.4f}  ->  Binary: {binary_preds[i][0]}  (True: {y[i][0]})"
        )

    #print overall prediction accuracy of network
    accuracy = np.mean(binary_preds == y) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    #print final loss after training
    print(f"Final loss: {loss_hist[-1]:.6f}")

    #generate and save plot displaying change in loss over time while training
    plt.figure(figsize=(10, 6))
    plt.plot(loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.grid(True)
    plt.savefig("xor_training_loss.png")
    print("\nLoss curve saved as 'xor_training_loss.png'")


if __name__ == "__main__":
    main()
