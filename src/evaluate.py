"""
Evaluation script for the NumPy character recognition neural network.

This module evaluates the trained model on the unseen test dataset.
It calculates test loss, test accuracy, and generates a confusion
matrix for detailed classification analysis.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork


# ============================================================
# Configuration
# ============================================================

PROCESSED_DATA_DIR = os.path.join(
    "data",
    "processed"
)

MODEL_PATH = os.path.join(
    "models",
    "character_network.npz"
)

RESULTS_DIR = "results"

CLASS_NAMES = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
) + list(
    "123456789"
)


# ============================================================
# Load Test Data
# ============================================================

def load_test_data():
    """
    Load the unseen test dataset.

    Returns
    -------
    tuple
        X_test and y_test arrays.
    """

    X_test = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_test.npy"
        )
    )

    return X_test, y_test


# ============================================================
# Load Trained Model
# ============================================================

def load_model():
    """
    Load the trained neural network parameters from disk.

    Returns
    -------
    NeuralNetwork
        Neural network containing the trained weights.
    """

    network = NeuralNetwork(
        input_size=784,
        hidden1_size=128,
        hidden2_size=64,
        output_size=35,
        learning_rate=0.01
    )

    model_data = np.load(
        MODEL_PATH
    )

    network.W1 = model_data["W1"]
    network.b1 = model_data["b1"]

    network.W2 = model_data["W2"]
    network.b2 = model_data["b2"]

    network.W3 = model_data["W3"]
    network.b3 = model_data["b3"]

    return network


# ============================================================
# Accuracy
# ============================================================

def calculate_accuracy(
    y_true,
    probabilities
):
    """
    Calculate classification accuracy.

    Parameters
    ----------
    y_true : numpy.ndarray
        Correct class labels.

    probabilities : numpy.ndarray
        Predicted probabilities for all classes.

    Returns
    -------
    float
        Classification accuracy.
    """

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return np.mean(
        predictions == y_true
    )


# ============================================================
# Confusion Matrix
# ============================================================

def create_confusion_matrix(
    y_true,
    y_pred,
    number_of_classes
):
    """
    Create a confusion matrix using NumPy.

    Rows represent actual classes and columns represent
    predicted classes.

    Parameters
    ----------
    y_true : numpy.ndarray
        Actual class labels.

    y_pred : numpy.ndarray
        Predicted class labels.

    number_of_classes : int
        Total number of classes.

    Returns
    -------
    numpy.ndarray
        Confusion matrix.
    """

    matrix = np.zeros(
        (
            number_of_classes,
            number_of_classes
        ),
        dtype=int
    )

    for actual, predicted in zip(
        y_true,
        y_pred
    ):
        matrix[
            actual,
            predicted
        ] += 1

    return matrix


# ============================================================
# Plot Confusion Matrix
# ============================================================

def plot_confusion_matrix(
    matrix
):
    """
    Plot and save the confusion matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Confusion matrix.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(
        matrix,
        interpolation="nearest"
    )

    plt.title(
        "Character Recognition Confusion Matrix"
    )

    plt.colorbar()

    tick_positions = np.arange(
        len(CLASS_NAMES)
    )

    plt.xticks(
        tick_positions,
        CLASS_NAMES,
        rotation=90
    )

    plt.yticks(
        tick_positions,
        CLASS_NAMES
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nConfusion matrix saved to:"
        f"\n{output_path}"
    )


# ============================================================
# Main Evaluation
# ============================================================

def main():
    """
    Evaluate the trained neural network on the test dataset.
    """

    print("=" * 70)
    print("CHARACTER RECOGNITION MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    print(
        f"\nTest samples: {X_test.shape[0]}"
    )

    print(
        f"Input features: {X_test.shape[1]}"
    )

    # --------------------------------------------------------
    # Load trained network
    # --------------------------------------------------------

    network = load_model()

    print(
        "\nTrained model loaded successfully."
    )

    # --------------------------------------------------------
    # Forward propagation
    # --------------------------------------------------------

    probabilities = network.forward(
        X_test
    )

    # --------------------------------------------------------
    # Test loss
    # --------------------------------------------------------

    test_loss = network.cross_entropy_loss(
        y_test,
        probabilities
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Test accuracy
    # --------------------------------------------------------

    test_accuracy = calculate_accuracy(
        y_test,
        probabilities
    )

    print(
        f"\nTest Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Accuracy Percentage: "
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion_matrix = create_confusion_matrix(
        y_test,
        predictions,
        len(CLASS_NAMES)
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix)

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    plot_confusion_matrix(
        confusion_matrix
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "y_test.npy"
        ),
        y_test
    )

    np.save(
        os.path.join(
            RESULTS_DIR,
            "test_predictions.npy"
        ),
        predictions
    )

    print(
        "\nEvaluation completed."
    )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()