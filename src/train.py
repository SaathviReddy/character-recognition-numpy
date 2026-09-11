import os
import numpy as np
from neural_network import NeuralNetwork
# ============================================================
# Configuration
# ============================================================

PROCESSED_DATA_DIR = os.path.join(
    "data",
    "processed"
)

EPOCHS = 200

BATCH_SIZE = 64

LEARNING_RATE = 0.05

RANDOM_SEED = 42


# ============================================================
# Data Loading
# ============================================================

def load_data():
    X_train = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_train.npy"
        )
    )

    y_train = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_train.npy"
        )
    )

    X_val = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_val.npy"
        )
    )

    y_val = np.load(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_val.npy"
        )
    )

    return X_train, y_train, X_val, y_val


# ============================================================
# Accuracy Calculation
# ============================================================

def calculate_accuracy(y_true, probabilities):
    predictions = np.argmax(
        probabilities,
        axis=1
    )
    accuracy = np.mean(
        predictions == y_true
    )

    return accuracy
# ============================================================
# Training
# ============================================================

def train_network(
    network,
    X_train,
    y_train,
    X_val,
    y_val
):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    number_of_samples = X_train.shape[0]

    for epoch in range(1, EPOCHS + 1):

        # ----------------------------------------------------
        # Shuffle training data
        # ----------------------------------------------------

        indices = np.arange(
            number_of_samples
        )

        rng.shuffle(indices)

        X_train_shuffled = X_train[
            indices
        ]

        y_train_shuffled = y_train[
            indices
        ]

        # ----------------------------------------------------
        # Mini-batch training
        # ----------------------------------------------------

        for start in range(
            0,
            number_of_samples,
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                number_of_samples
            )

            X_batch = X_train_shuffled[
                start:end
            ]

            y_batch = y_train_shuffled[
                start:end
            ]

            # Forward propagation.
            network.forward(
                X_batch
            )

            # Backpropagation.
            gradients = network.backward(
                X_batch,
                y_batch
            )

            (
                dW1,
                db1,
                dW2,
                db2,
                dW3,
                db3
            ) = gradients

            # Gradient descent.
            network.update_parameters(
                dW1,
                db1,
                dW2,
                db2,
                dW3,
                db3
            )

        # ----------------------------------------------------
        # Training metrics
        # ----------------------------------------------------

        train_probabilities = network.forward(
            X_train
        )

        train_loss = network.cross_entropy_loss(
            y_train,
            train_probabilities
        )

        train_accuracy = calculate_accuracy(
            y_train,
            train_probabilities
        )

        # ----------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------

        val_probabilities = network.forward(
            X_val
        )

        val_loss = network.cross_entropy_loss(
            y_val,
            val_probabilities
        )

        val_accuracy = calculate_accuracy(
            y_val,
            val_probabilities
        )

        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    return history


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("CHARACTER RECOGNITION NEURAL NETWORK TRAINING")
    print("=" * 70)

    # Load datasets.
    (
        X_train,
        y_train,
        X_val,
        y_val
    ) = load_data()

    print("\nDataset information:")
    print(
        f"Training samples: {X_train.shape[0]}"
    )

    print(
        f"Validation samples: {X_val.shape[0]}"
    )

    print(
        f"Input features: {X_train.shape[1]}"
    )

    print(
        f"Number of classes: "
        f"{len(np.unique(y_train))}"
    )

    # Create neural network.
    network = NeuralNetwork(
        input_size=784,
        hidden1_size=256,
        hidden2_size=128,
        output_size=35,
        learning_rate=LEARNING_RATE
    )

    print("\nNetwork created.")

    print(
        "\nTraining started...\n"
    )

    # Train network.
    history = train_network(
        network,
        X_train,
        y_train,
        X_val,
        y_val
    )

    # Save trained parameters.
    os.makedirs(
        "models",
        exist_ok=True
    )

    np.savez(
        os.path.join(
            "models",
            "character_network.npz"
        ),
        W1=network.W1,
        b1=network.b1,
        W2=network.W2,
        b2=network.b2,
        W3=network.W3,
        b3=network.b3
    )

    # Save training history.
    np.save(
        os.path.join(
            "models",
            "training_history.npy"
        ),
        history
    )

    print("\nTraining completed.")

    print(
        "\nFinal training accuracy: "
        f"{history['train_accuracy'][-1]:.4f}"
    )

    print(
        "Final validation accuracy: "
        f"{history['val_accuracy'][-1]:.4f}"
    )

    print(
        "\nModel saved to:"
        "\nmodels/character_network.npz"
    )
# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()