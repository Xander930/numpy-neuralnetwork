import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source.network import NeuralNet

def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    print("\nTraining Data")
    for i in range(len(X)):
        print(f"    Input: {X[i]} -> Output: {y[i][0]}")

    print("Beginning Training...")

    nn = NeuralNet(
        size_in=2,
        size_hidden=4,
        size_out=1,
        learning_rate=0.5
    )

    loss_hist = nn.train(X, y, epochs=10000, verbose=True)

    print("Results: ")

    predictions = nn.forward(X)
    binary_preds = nn.predict(X, threshold=0.5)

    print("\nPredictions: ")
    for i in range(len(X)):
        print(f"  Input: {X[i]}  ->  Predicted: {predictions[i][0]:.4f}  ->  Binary: {binary_preds[i][0]}  (True: {y[i][0]})")

    accuracy = np.mean(binary_preds == y) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

    print(f"Final loss: {loss_hist[-1]:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_hist)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    plt.savefig('xor_training_loss.png')
    print("\nLoss curve saved as 'xor_training_loss.png'")