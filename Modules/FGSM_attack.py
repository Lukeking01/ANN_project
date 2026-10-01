import numpy as np
from Modules.ANN import Network


class FGSMAdversary():
    """
    Takes in a starting image x with true output y, and changes it 
    in a single step in the direction that most increases the model's loss.
    """
    def __init__(self, net: Network, x, y):
        self.net = net
        self.x = x
        self.y = y

    def _classify(self, x_k):
        """
        Returns true if x_k is classified as the same output class as the original.
        """
        x_k = np.atleast_2d(x_k)
        y_k = self.net.predict_batch(x_k)
        y_true = np.atleast_2d(self.y)
        return np.argmax(y_k, axis=1) == np.argmax(y_true, axis=1)

    def _input_gradient(self, x, y):
        """
        Computes the gradient of the loss with respect to the input x, by running a forward
        pass and then backpropagating the error one step further than training does, through
        the first weight matrix, to reach the input itself.
        """
        A = np.asarray(x, dtype=float)
        if A.ndim == 1:
            A = A.reshape(1, -1)

        Y = np.asarray(y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        activations = [A]
        preactivations = []

        # Step 1. Forward pass, keeping every pre-activation for the backward pass.
        for W, b, act in zip(self.net.weights, self.net.biases, self.net.activation_funcs[1:]):
            Z = A @ W.T + b
            preactivations.append(Z)
            A = act(Z)
            activations.append(A)

        # Step 2. Output-layer delta, same formula the network uses for its own weight gradients.
        delta = (
            self.net.loss_derivative(activations[-1], Y)
            * self.net.activation_derivatives[-1](preactivations[-1])
        )

        # Backpropagate delta through every hidden layer.
        for layer_index in range(len(self.net.weights) - 2, -1, -1):
            delta = (
                delta @ self.net.weights[layer_index + 1]
                * self.net.activation_derivatives[layer_index + 1](
                    preactivations[layer_index]
                )
            )

        # One last step back through the first weight matrix reaches the input itself.
        grad_x = delta @ self.net.weights[0]
        return grad_x

    def generate(self, eps=0.15, verbose=False):
        """
        Main attacking method. Generates a new image via a single FGSM step:
            x_adv = x + eps * sign(grad_x Loss(f(x), y))
        """

        # Step 2 and 3. Gradient of the loss w.r.t. the input, then a single signed step.
        grad_x = self._input_gradient(self.x, self.y)
        x_adv = np.atleast_2d(self.x) + eps * np.sign(grad_x)

        # Clip candidate to ensure it's in the output range.
        x_adv = np.clip(x_adv, 0.0, 1.0)

        if verbose:
            still_correct = self._classify(x_adv)
            success_rate = np.mean(~still_correct)
            print(f"eps={eps}: {success_rate * 100:.2f}% of the batch misclassified "
                  f"({np.sum(~still_correct)}/{len(still_correct)})")

        # Preserve the caller's original shape.
        if np.asarray(self.x).ndim == 1:
            return x_adv[0]
        return x_adv