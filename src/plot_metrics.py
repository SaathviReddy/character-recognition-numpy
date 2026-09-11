import os
import numpy as np
import matplotlib.pyplot as plt
# ============================================================
# Configuration
# ============================================================

HISTORY_PATH = os.path.join(
    "models",
    "training_history.npy"
)

RESULTS_DIR = "results"
# ============================================================
# Load Training History
# ============================================================

def load_history():
    """
    Load training and validation metrics from disk.

    Returns
    -------
    dict
        Dictionary containing training loss, training accuracy,
        validation loss, and validation accuracy.
    """

    history = np.load(
        HISTORY_PATH,
        allow_pickle=True
    ).item()

    return history


# ============================================================
# Plot Accuracy
# ============================================================

def plot_accuracy(history):
    """
    Create and save the training and validation accuracy curve.

    Parameters
    ----------
    history : dict
        Training history containing accuracy values.
    """

    epochs = range(
        1,
        len(history["train_accuracy"]) + 1
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "accuracy_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Accuracy graph saved to:\n{output_path}"
    )


# ============================================================
# Plot Loss
# ============================================================

def plot_loss(history):
    """
    Create and save the training and validation loss curve.

    Parameters
    ----------
    history : dict
        Training history containing loss values.
    """

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Cross-Entropy Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    output_path = os.path.join(
        RESULTS_DIR,
        "loss_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Loss graph saved to:\n{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():
    """
    Load the saved training history and generate metric plots.
    """

    print("=" * 70)
    print("TRAINING METRICS VISUALIZATION")
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    history = load_history()

    print(
        f"\nNumber of epochs: "
        f"{len(history['train_loss'])}"
    )

    plot_accuracy(
        history
    )

    plot_loss(
        history
    )

    print(
        "\nMetric visualization completed."
    )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()
