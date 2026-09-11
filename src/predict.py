import os
import sys
import numpy as np
from PIL import Image


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "character_network.npz"
)

IMAGE_SIZE = 28

CLASS_NAMES = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
) + list(
    "123456789"
)
# ============================================================
# Activation Functions
# ============================================================

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )


# ============================================================
# Load Model
# ============================================================

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = np.load(MODEL_PATH)

    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]
    W3 = model["W3"]
    b3 = model["b3"]

    return W1, b1, W2, b2, W3, b3


# ============================================================
# Image Preprocessing
# ============================================================

def preprocess_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path)

    # Convert to grayscale.
    image = image.convert("L")

    # Resize to 28 x 28.
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert to NumPy array.
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize pixel values.
    image_array = image_array / 255.0

    # Flatten 28 x 28 into 784 features.
    image_array = image_array.reshape(
        1,
        784
    )

    return image_array


# ============================================================
# Prediction
# ============================================================

def predict(image_path):

    W1, b1, W2, b2, W3, b3 = load_model()

    X = preprocess_image(
        image_path
    )

    # Forward propagation.
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, W3) + b3
    probabilities = softmax(Z3)

    # Find highest probability.
    predicted_index = np.argmax(
        probabilities[0]
    )

    predicted_character = CLASS_NAMES[
        predicted_index
    ]

    confidence = probabilities[
        0,
        predicted_index
    ]

    return predicted_character, confidence


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("CHARACTER RECOGNITION - SINGLE IMAGE PREDICTION")
    print("=" * 70)

    # Get image path from command line.
    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            "python src\\predict.py <image_path>"
        )

        print("\nExample:")
        print(
            "python src\\predict.py data\\raw\\A\\A_001.png"
        )

        return

    image_path = sys.argv[1]

    print(
        f"\nInput image: {image_path}"
    )

    try:

        predicted_character, confidence = predict(
            image_path
        )

        print("\nPrediction:")
        print(
            f"Character: {predicted_character}"
        )

        print(
            f"Confidence: {confidence:.2%}"
        )

        print("\nPrediction completed.")

    except Exception as error:

        print(
            f"\nError: {error}"
        )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()