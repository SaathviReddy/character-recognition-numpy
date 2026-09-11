import os
import numpy as np
from PIL import Image
# ============================================================
# Configuration
# ============================================================

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("123456789")
IMAGE_SIZE = 28
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
# ============================================================
# Image Loading
# ============================================================

def load_dataset():
    """
    Load all character images from the raw dataset.

    Each image is converted to grayscale, resized to 28x28,
    converted into a NumPy array, normalized to the range
    0-1, and flattened into 784 input features.

    Returns
    -------
    X : numpy.ndarray
        Image data with shape (number_of_images, 784).

    y : numpy.ndarray
        Integer class labels with shape (number_of_images,).
    """

    images = []
    labels = []

    print("Loading images...")

    for label, character in enumerate(CLASSES):

        class_directory = os.path.join(
            RAW_DATA_DIR,
            character
        )

        if not os.path.isdir(class_directory):
            raise FileNotFoundError(
                f"Class directory not found: {class_directory}"
            )

        image_files = sorted(
            file_name
            for file_name in os.listdir(class_directory)
            if file_name.lower().endswith(".png")
        )

        print(
            f"Loading class {character}: "
            f"{len(image_files)} images"
        )

        for file_name in image_files:

            image_path = os.path.join(
                class_directory,
                file_name
            )

            # Open the image and convert it to grayscale.
            image = Image.open(image_path).convert("L")

            # Ensure every image is 28x28 pixels.
            image = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            # Convert image pixels to a NumPy array.
            image_array = np.array(
                image,
                dtype=np.float32
            )

            # Normalize pixel values from 0-255 to 0-1.
            image_array = image_array / 255.0

            # Flatten 28x28 into 784 features.
            image_array = image_array.reshape(-1)

            images.append(image_array)

            labels.append(label)

    X = np.array(images, dtype=np.float32)

    y = np.array(labels, dtype=np.int64)

    return X, y


# ============================================================
# Dataset Splitting
# ============================================================
def split_dataset(X, y):
    """
    Shuffle and split the dataset into training, validation,
    and testing sets while maintaining equal class distribution.

    Each class contributes:
        70% training
        15% validation
        15% testing
    """

    rng = np.random.default_rng(RANDOM_SEED)

    train_indices = []
    validation_indices = []
    test_indices = []

    # Process each class separately.
    for class_label in range(len(CLASSES)):

        # Find all images belonging to this class.
        class_indices = np.where(
            y == class_label
        )[0]

        # Shuffle the class-specific indices.
        rng.shuffle(class_indices)

        number_of_samples = len(class_indices)

        train_end = int(
            number_of_samples * TRAIN_RATIO
        )

        validation_end = train_end + int(
            number_of_samples * VALIDATION_RATIO
        )

        train_indices.extend(
            class_indices[:train_end]
        )

        validation_indices.extend(
            class_indices[
                train_end:validation_end
            ]
        )

        test_indices.extend(
            class_indices[validation_end:]
        )

    # Shuffle each final split.
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)

    # Create datasets.
    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[validation_indices]
    y_val = y[validation_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )
# ============================================================
# Save Processed Dataset
# ============================================================

def save_dataset(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    """
    Save all processed datasets as NumPy .npy files.

    The files are stored inside data/processed/.
    """

    os.makedirs(
        PROCESSED_DATA_DIR,
        exist_ok=True
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_train.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_val.npy"
        ),
        X_val
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "X_test.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_val.npy"
        ),
        y_val
    )

    np.save(
        os.path.join(
            PROCESSED_DATA_DIR,
            "y_test.npy"
        ),
        y_test
    )

    print("\nProcessed datasets saved successfully.")


# ============================================================
# Main Processing Pipeline
# ============================================================

def main():
    """
    Execute the complete preprocessing pipeline.
    """

    print("=" * 60)
    print("CHARACTER DATASET PREPROCESSING")
    print("=" * 60)

    # Step 1: Load and preprocess images.
    X, y = load_dataset()

    print("\nComplete dataset shape:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Step 2: Shuffle and split the dataset.
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_dataset(X, y)
    print("\nDataset split:")
    print(f"Training:   {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Testing:    {X_test.shape[0]} samples")
    # Step 3: Display normalization information.
    print("\nPixel value range:")
    print(f"Minimum: {X_train.min():.4f}")
    print(f"Maximum: {X_train.max():.4f}")
    # Step 4: Save the processed data.
    save_dataset(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )
    print("\nPreprocessing completed successfully.")
# ============================================================
# Program Entry Point
# ============================================================
if __name__ == "__main__":
    main()