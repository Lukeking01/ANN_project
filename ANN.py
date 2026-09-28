import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Activation functions
# -------------------------

def step(z):
    return z > 0


def step_derivative(z):
    return 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def linear(z):
    return z


def linear_derivative(z):
    return 1.0

def mish(z):
    return z*np.tanh(np.log(1+np.exp(z)))

def mish_derivative(z):
    return np.tanh(np.log(1+np.exp(z))) + z/np.cosh(np.log(1+np.exp(z)))**2 * sigmoid(z)
# -------------------------
# Loss functions
# -------------------------

def mse_loss(output, target):
    return 0.5 * np.sum((output - target) ** 2)


def binary_cross_entropy(output, target):
    epsilon = 1e-12
    output = np.clip(output, epsilon, 1 - epsilon)

    return -np.sum(
        target * np.log(output)
        + (1 - target) * np.log(1 - output)
    )


def mse_derivative(output, target):
    return output - target


def binary_cross_entropy_derivative(output, target):
    epsilon = 1e-12
    output = np.clip(output, epsilon, 1 - epsilon)

    return (output - target) / (output * (1 - output))


def one_hot(label, num_classes=10):
    result = np.zeros(num_classes)
    result[label] = 1
    return result

ACTIVATIONS = {
    "step": (step, step_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
    "linear": (linear, linear_derivative),
    "mish": (mish, mish_derivative)
}

LOSSES = {
    "mse": (mse_loss, mse_derivative),
    "binary_cross_entropy": (
        binary_cross_entropy,
        binary_cross_entropy_derivative,
    ),
}

class Network:

    def __init__(
        self,
        structure: list[int],
        activation_functions : str,
        activation_derivatives = None,
        loss_function: str="mse",
        loss_derivative = None
    ):
        """Matrix-based neural network.

        X data has shape (number_of_samples, input_size).
        Y data has shape (number_of_samples, output_size).
        """

        if len(structure) != len(activation_functions):
            raise ValueError(
                "structure and activation_funcs must have the same length"
            )


        self.structure = structure
        self.activation_funcs = [ACTIVATIONS[func][0] for func in activation_functions]
        self.activation_derivatives = [ACTIVATIONS[func][1] for func in activation_functions]
        self.loss_function = LOSSES[loss_function][0]
        self.loss_derivative = LOSSES[loss_function][1]
        
        if activation_derivatives:
            self.activation_derivatives = activation_derivatives
        if loss_derivative:
            self.loss_derivative = loss_derivative

        self.weights = []
        self.biases = []

        for i in range(1, len(structure)):
            previous_size = structure[i - 1]

            self.weights.append(
                np.random.randn(structure[i], previous_size)
                * np.sqrt(2 / previous_size)
            )

            self.biases.append(np.zeros(structure[i]))

    # ---------------------------------------------------------
    # Forward propagation
    # ---------------------------------------------------------

    def _forward_batch(self, X):
        """Forward propagation for many samples at once."""

        A = np.asarray(X, dtype=float)

        for W, b, activation_function in zip(
            self.weights,
            self.biases,
            self.activation_funcs[1:]
        ):
            A = activation_function(A @ W.T + b)

        return A

    def forward(self, inputs):
        """Forward propagation for one sample."""

        X = np.asarray(inputs, dtype=float).reshape(1, -1)
        return self._forward_batch(X)[0]

    # ---------------------------------------------------------
    # Backpropagation
    # ---------------------------------------------------------

    def _batch_gradients(self, X, Y):
        """Calculate gradients for an entire mini-batch at once."""

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        activations = [X]
        preactivations = []

        A = X

        for W, b, activation_function in zip(
            self.weights,
            self.biases,
            self.activation_funcs[1:]
        ):
            Z = A @ W.T + b
            preactivations.append(Z)
            A = activation_function(Z)
            activations.append(A)

        delta = (
            self.loss_derivative(activations[-1], Y)
            * self.activation_derivatives[-1](preactivations[-1])
        )

        deltas = [delta]

        for layer_index in range(len(self.weights) - 2, -1, -1):
            delta = (
                deltas[0] @ self.weights[layer_index + 1]
                * self.activation_derivatives[layer_index + 1](
                    preactivations[layer_index]
                )
            )

            deltas.insert(0, delta)

        weight_gradients = []
        bias_gradients = []

        for i, delta in enumerate(deltas):
            weight_gradients.append(delta.T @ activations[i])
            bias_gradients.append(np.sum(delta, axis=0))

        return weight_gradients, bias_gradients

    # ---------------------------------------------------------
    # Mini-batch update
    # ---------------------------------------------------------

    def update_mini_batch(self, X_batch, Y_batch, learning_rate):
        """Perform one gradient-descent update."""

        weight_gradients, bias_gradients = self._batch_gradients(
            X_batch,
            Y_batch
        )

        batch_size = len(X_batch)

        for i in range(len(self.weights)):
            self.weights[i] -= (
                learning_rate * weight_gradients[i] / batch_size
            )

            self.biases[i] -= (
                learning_rate * bias_gradients[i] / batch_size
            )

    # ---------------------------------------------------------
    # Weight initialization
    # ---------------------------------------------------------

    def randomize_weights(self):
        """Reinitialize all weights and biases."""

        for i in range(len(self.weights)):
            previous_size = self.structure[i]

            self.weights[i] = (
                np.random.randn(
                    self.structure[i + 1],
                    previous_size
                )
                * np.sqrt(2 / previous_size)
            )

            self.biases[i].fill(0)

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    def calculate_loss(self, X, Y):
        """Calculate average loss over a dataset."""

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        outputs = self._forward_batch(X)

        losses = np.array([
            self.loss_function(output, target)
            for output, target in zip(outputs, Y)
        ])

        return np.mean(losses)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def evaluate(self, X, Y):
        """Calculate classification accuracy.

        Y can be one-hot encoded with shape (N, classes)
        or integer labels with shape (N,).
        """

        predictions = np.argmax(self._forward_batch(X), axis=1)

        Y = np.asarray(Y)

        if Y.ndim > 1:
            actual = np.argmax(Y, axis=1)
        else:
            actual = Y

        return np.mean(predictions == actual)

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    def confusion_matrix(self, X, Y, nr_labels, labels=True):
        """Display and return a confusion matrix."""

        predictions = np.argmax(self._forward_batch(X), axis=1)

        Y = np.asarray(Y)

        if Y.ndim > 1:
            actual = np.argmax(Y, axis=1)
        else:
            actual = Y

        matrix = np.zeros((nr_labels, nr_labels), dtype=int)

        # Rows = predicted, columns = actual
        np.add.at(matrix, (predictions, actual), 1)

        fig, ax = plt.subplots(figsize=(nr_labels, nr_labels))
        ax.imshow(matrix)

        if labels:
            for i in range(nr_labels):
                for j in range(nr_labels):
                    ax.text(
                        j, i, matrix[i, j],
                        ha="center",
                        va="center"
                    )

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_xticks(np.linspace(0,nr_labels-1,nr_labels))
        ax.set_yticks(np.linspace(0,nr_labels-1,nr_labels))
        ax.set_title("Confusion Matrix")

        plt.show()

        return matrix

    # ---------------------------------------------------------
    # Stochastic Gradient Descent
    # ---------------------------------------------------------

    def SGD(
        self,
        X_train,
        Y_train,
        mini_batch_size=1,
        learning_rate=0.1,
        X_val=None,
        Y_val=None,
        tolerance=1e-4,
        verbose=True,
        max_epoch=2e4,
        step_tol=1e-5,
        learning_rate_decay = 0
    ):
        """Train using mini-batch stochastic gradient descent.

        Training data is passed separately:
            X_train = input samples
            Y_train = targets

        Validation data is passed separately:
            X_val, Y_val
        """

        X_train = np.asarray(X_train, dtype=float)
        Y_train = np.asarray(Y_train, dtype=float)

        if len(X_train) != len(Y_train):
            raise ValueError(
                "X_train and Y_train must have the same number of samples"
            )

        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty")

        if mini_batch_size <= 0:
            raise ValueError(
                "mini_batch_size must be greater than 0"
            )

        epoch = 0
        previous_loss = 100.0

        training_losses = []
        validation_losses = []

        while True:

            # Shuffle both arrays using the same permutation.
            indices = np.random.permutation(len(X_train))

            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            # Train on each mini-batch.
            for start in range(0, len(X_train), mini_batch_size):

                end = start + mini_batch_size

                self.update_mini_batch(
                    X_shuffled[start:end],
                    Y_shuffled[start:end],
                    learning_rate
                )

            # Training loss.
            loss = self.calculate_loss(
                X_train,
                Y_train
            )

            training_losses.append(loss)

            # Validation.
            if X_val is not None and Y_val is not None:

                accuracy = self.evaluate(
                    X_val,
                    Y_val
                )

                val_loss = self.calculate_loss(
                    X_val,
                    Y_val
                )

                validation_losses.append(val_loss)

                if verbose:
                    print(
                        f"Epoch {epoch + 1}"
                        f"  Training loss: {loss:.4e}"
                        f"  Validation loss: {val_loss:.4e}"
                        f"  Validation accuracy: "
                        f"{accuracy * 100:.2f}%"
                    )

            elif verbose:
                print(
                    f"Epoch {epoch + 1}"
                    f"  Loss: {loss:.5e} --> {tolerance}"
                )

            epoch += 1

            if loss < tolerance or epoch + 1 > max_epoch:
                return (
                    validation_losses,
                    training_losses,
                    loss,
                    epoch
                )

            if np.abs(loss - previous_loss) < step_tol:
                return (
                    validation_losses,
                    training_losses,
                    loss,
                    epoch
                )

            previous_loss = loss
            
            if learning_rate_decay:
                learning_rate *= learning_rate_decay

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    def predict(self, inputs):
        """Make a prediction for one sample."""

        return self.forward(inputs)

    def predict_batch(self, X):
        """Make predictions for many samples at once."""

        return self._forward_batch(X)

    # ---------------------------------------------------------
    # Saving and loading
    # ---------------------------------------------------------

    def save(self, filename):
        """Save the complete network."""

        activation_names = [
            next(
                name for name, (func, _) in ACTIVATIONS.items()
                if func is activation
            )
            for activation in self.activation_funcs
        ]

        loss_name = next(
            name
            for name, (func, _) in LOSSES.items()
            if func is self.loss_function
        )

        np.savez_compressed(
            filename,
            structure=np.array(self.structure),
            weights=np.array(self.weights, dtype=object),
            biases=np.array(self.biases, dtype=object),
            activation_names=np.array(activation_names),
            loss_name=loss_name,
        )

    @classmethod
    def load(cls, filename):
        """Load a complete network from a saved file."""

        data = np.load(
            filename,
            allow_pickle=True
        )

        structure = data["structure"].tolist()

        activation_names = data["activation_names"].tolist()

        activation_functions = activation_names

        activation_derivatives = [
            ACTIVATIONS[name][1]
            for name in activation_names
        ]

        loss_name = str(data["loss_name"])
        

        network = cls(
            structure=structure,
            activation_functions=activation_functions,
            activation_derivatives=activation_derivatives,
            loss_function=loss_name,
        )

        network.weights = list(data["weights"])
        network.biases = list(data["biases"])

        return network
