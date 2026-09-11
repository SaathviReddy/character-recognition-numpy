"""
Analyze incorrect predictions made by the trained character
recognition neural network.

The script identifies misclassified test images and displays
five real examples with their actual and predicted classes.
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

CLASS_NAMES = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
) + list(
    "123456789"
)


# ============================================================
# Load Model
# ============================================================

def load_model():
    """
    Load the trained neural network parameters.

    Returns
    -------
    NeuralNetwork
        Trained neural network.
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
# Main
# ============================================================

def main():
    """
    Find and display five incorrect test predictions.
    """

    print("=" * 70)
    print("INCORRECT PREDICTION ANALYSIS")
    print("=" * 70)

    # Load test data.
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

    # Load trained model.
    network = load_model()

    # Generate predictions.
    probabilities = network.forward(
        X_test
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    # Find incorrect predictions.
    incorrect_indices = np.where(
        predictions != y_test
    )[0]

    print(
        f"\nTotal incorrect predictions: "
        f"{len(incorrect_indices)}"
    )

    # Select first five incorrect examples.
    selected_indices = incorrect_indices[:5]

    print("\nFive incorrect predictions:\n")

    for number, index in enumerate(
        selected_indices,
        start=1
    ):

        actual_class = CLASS_NAMES[
            y_test[index]
        ]

        predicted_class = CLASS_NAMES[
            predictions[index]
        ]

        confidence = probabilities[
            index,
            predictions[index]
        ]

        print(
            f"{number}. "
            f"Actual = {actual_class}, "
            f"Predicted = {predicted_class}, "
            f"Confidence = {confidence:.4f}"
        )

    # --------------------------------------------------------
    # Display images
    # --------------------------------------------------------

    plt.figure(
        figsize=(15, 3)
    )

    for position, index in enumerate(
        selected_indices
    ):

        plt.subplot(
            1,
            5,
            position + 1
        )

        image = X_test[index].reshape(
            28,
            28
        )

        plt.imshow(
            image,
            cmap="gray"
        )

        actual_class = CLASS_NAMES[
            y_test[index]
        ]

        predicted_class = CLASS_NAMES[
            predictions[index]
        ]

        confidence = probabilities[
            index,
            predictions[index]
        ]

        plt.title(
            f"Actual: {actual_class}\n"
            f"Pred: {predicted_class}\n"
            f"Conf: {confidence:.2f}"
        )

        plt.axis("off")

    plt.tight_layout()

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = os.path.join(
        "results",
        "incorrect_predictions.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.show()

    print(
        f"\nIncorrect prediction image saved to:"
        f"\n{output_path}"
    )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()