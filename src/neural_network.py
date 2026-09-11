import numpy as np
class NeuralNetwork:
    def __init__(
        self,
        input_size=784,
        hidden1_size=128,
        hidden2_size=64,
        output_size=35,
        learning_rate=0.01
    ):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # ----------------------------------------------------
        # Reproducible weight initialization
        # ----------------------------------------------------

        rng = np.random.default_rng(42)

        # Input → Hidden Layer 1
        self.W1 = (
            rng.standard_normal(
                (input_size, hidden1_size)
            )
            * np.sqrt(2.0 / input_size)
        )

        self.b1 = np.zeros(
            (1, hidden1_size)
        )

        # Hidden Layer 1 → Hidden Layer 2
        self.W2 = (
            rng.standard_normal(
                (hidden1_size, hidden2_size)
            )
            * np.sqrt(2.0 / hidden1_size)
        )

        self.b2 = np.zeros(
            (1, hidden2_size)
        )

        # Hidden Layer 2 → Output Layer
        self.W3 = (
            rng.standard_normal(
                (hidden2_size, output_size)
            )
            * np.sqrt(2.0 / hidden2_size)
        )

        self.b3 = np.zeros(
            (1, output_size)
        )
            # ========================================================
    # ReLU Activation
    # ========================================================

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    # ========================================================
    # ReLU Derivative
    # ========================================================

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
        # ========================================================
    # Softmax Activation
    # ========================================================

    @staticmethod
    def softmax(x):
        # Subtract the maximum value for numerical stability.
        shifted_x = x - np.max(
            x,
            axis=1,
            keepdims=True
        )

        # Calculate exponentials.
        exp_values = np.exp(shifted_x)

        # Normalize so probabilities sum to 1.
        probabilities = (
            exp_values
            / np.sum(
                exp_values,
                axis=1,
                keepdims=True
            )
        )

        return probabilities
        # ========================================================
    # Cross-Entropy Loss
    # ========================================================

    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        number_of_samples = y_true.shape[0]

        # Prevent log(0), which would produce numerical errors.
        probabilities = np.clip(
            y_pred,
            1e-12,
            1.0 - 1e-12
        )

        # Select the probability assigned to the correct class.
        correct_class_probabilities = probabilities[
            np.arange(number_of_samples),
            y_true
        ]

        # Calculate average negative log likelihood.
        loss = -np.mean(
            np.log(correct_class_probabilities)
        )

        return loss
        # ========================================================
    # Forward Propagation
    # ========================================================

    def forward(self, X):
        # ----------------------------------------------------
        # Layer 1
        # ----------------------------------------------------

        # Weighted sum:
        # Z1 = XW1 + b1
        self.Z1 = X @ self.W1 + self.b1

        # Apply ReLU activation.
        self.A1 = self.relu(self.Z1)

        # ----------------------------------------------------
        # Layer 2
        # ----------------------------------------------------

        # Weighted sum:
        # Z2 = A1W2 + b2
        self.Z2 = self.A1 @ self.W2 + self.b2

        # Apply ReLU activation.
        self.A2 = self.relu(self.Z2)

        # ----------------------------------------------------
        # Output Layer
        # ----------------------------------------------------

        # Calculate output logits.
        self.Z3 = self.A2 @ self.W3 + self.b3

        # Convert logits into probabilities.
        self.A3 = self.softmax(self.Z3)

        return self.A3
        # ========================================================
    # Backpropagation
    # ========================================================

    def backward(self, X, y):
        number_of_samples = X.shape[0]

        # ----------------------------------------------------
        # Output Layer Gradient
        # ----------------------------------------------------

        # For Softmax + Cross-Entropy, the derivative simplifies
        # to predicted probabilities minus the one-hot labels.
        dZ3 = self.A3.copy()

        dZ3[
            np.arange(number_of_samples),
            y
        ] -= 1

        dZ3 /= number_of_samples

        # Gradient of W3.
        dW3 = self.A2.T @ dZ3

        # Gradient of b3.
        db3 = np.sum(
            dZ3,
            axis=0,
            keepdims=True
        )

        # ----------------------------------------------------
        # Hidden Layer 2 Gradient
        # ----------------------------------------------------

        dA2 = dZ3 @ self.W3.T

        dZ2 = (
            dA2
            * self.relu_derivative(self.Z2)
        )

        dW2 = self.A1.T @ dZ2

        db2 = np.sum(
            dZ2,
            axis=0,
            keepdims=True
        )

        # ----------------------------------------------------
        # Hidden Layer 1 Gradient
        # ----------------------------------------------------

        dA1 = dZ2 @ self.W2.T

        dZ1 = (
            dA1
            * self.relu_derivative(self.Z1)
        )

        dW1 = X.T @ dZ1

        db1 = np.sum(
            dZ1,
            axis=0,
            keepdims=True
        )

        return (
            dW1,
            db1,
            dW2,
            db2,
            dW3,
            db3
        )
        # ========================================================
    # Gradient Descent
    # ========================================================

    def update_parameters(
        self,
        dW1,
        db1,
        dW2,
        db2,
        dW3,
        db3
    ):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
if __name__ == "__main__":
    # Create network.
    network = NeuralNetwork()

    # Create a small fake batch.
    X_test = np.random.rand(4, 784)

    # Four fake class labels.
    y_test = np.array([0, 5, 10, 20])
    # Forward propagation.
    predictions = network.forward(X_test)
    # Calculate loss.
    loss = network.cross_entropy_loss(
        y_test,
        predictions
    )
    # Backpropagation.
    gradients = network.backward(
        X_test,
        y_test
    )
    (
        dW1,
        db1,
        dW2,
        db2,
        dW3,
        db3
    ) = gradients
    print("Loss:")
    print(loss)
    print("\nGradient shapes:")
    print("dW1:", dW1.shape)
    print("db1:", db1.shape)
    print("dW2:", dW2.shape)
    print("db2:", db2.shape)
    print("dW3:", dW3.shape)
    print("db3:", db3.shape)