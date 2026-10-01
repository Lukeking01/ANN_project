#########################################################################################################
# Approach: Based on the paper: https://arxiv.org/pdf/1712.04248
# DECISION-BASED ADVERSARIAL ATTACKS: RELIABLE ATTACKS AGAINST BLACK-BOX MACHINE LEARNING MODELS
# Wieland Brendel, Jonas Rauber & Matthias Bethge
#
# 1. Pick an image to use as reference. Denote this x. It should be classified correctly, f(x) = y. ~x_k 
#    will be the adversarially perturbed image at the k-th step of the attack.
# 2. Pick a point, can be random noise, which is already adversarial to x. f(~x_0) != y.
#   2a. For instance, random sampling on a uniform distribution of pixel values from [0, 1].
# 3. For up to k_max iterations, perform a random walk along the boundary between the adversarial and 
#    non-adversarial region.
#   3a*. Each step is controlled by two hyper parameters. The size of the total perterbation "delta", 
#        and the reduced distance between the adversary and the original image after the step, "epsilon".
#########################################################################################################

import numpy as np
from Modules.ANN import Network
from Modules.data_io import save_points

class BlackBoxAdversary():
    # Takes in a starting image which we wish to mock with a new image that is as close as possible to the original
    # (in the L2 norm) but misclassified.
    def __init__(self, net: Network, x, y):
        self.net = net
        
        self.x = x
        self.y = y

    def _classify(self, x_k):
        """Returns true if x_k is classified as the same output class as the original."""
        y_k = self.net.predict(x_k)
        return np.argmax(y_k) == np.argmax(self.y)

    def _dist(self, x_k):
        """Returns the distance in the 2-norm between x and x_k."""
        return np.linalg.norm(self.x - x_k, ord=2)

    def generate(self, max_steps=50_000, image_count: int = None, verbose=False):
        """
        Main attacking method, generates a new image as close to the original "x" provided
        as possible while keeping it misclassified.

        :param max_steps: The total number of iterations to run.
        :param image_count: Maximum amount of images to save. Set to None to not save any images.
        x_k will be saved if there is a 20% or greater change between image distances.
        """

        # Hyper-parameters (adjusted dynamically)
        delta = 0.01
        epsilon = 0.0001
        # Stop when epsilon gets below the threshold
        epsilon_thres = 1e-8

        # Track the successful movements.
        successes = 0
        # Thresholds to increase/decrease the hyperparameters
        prop_thres_high = 0.50
        prop_thres_low = 0.40

        closeness_threshold = 1e-5

        # Step 1. Generate an image of random noise that is misclassified.
        x_k = np.random.random(self.x.shape)
        while self._classify(x_k):
            x_k = np.random.random(self.x.shape)

        # Store image points to display later
        xk_points = []
        dist = 0
        prev_dist = 1

        steps_taken = max_steps

        for step in range(max_steps):
            diff_v = self.x - x_k   # Vector x - x_k
            diff_v = diff_v / np.linalg.norm(diff_v)
            dist = self._dist(x_k)  # Distance between x, x_k in 2-norm

            # Logging
            if verbose:
                if step % 1000 == 0:
                    print(f"Step {step}: {round(dist, 5)}")
            # Exit condition
            if dist < closeness_threshold or epsilon < epsilon_thres:
                steps_taken = step
                break

            # Save image, note ignore offsets if it would save over 50 images
            if image_count and len(xk_points) < image_count and abs(prev_dist - dist) / prev_dist > 0.2:
                xk_points.append(x_k)
                prev_dist = dist

            # Step 2. Sample a random step, and project it orthogonally onto a hyper-sphere centered 
            # around x with radius ||x - x_k||.
            noise = np.random.randn(*x_k.shape)
            proj = np.dot(noise, diff_v) / (dist ** 2) * diff_v
            noise_orth = noise - proj

            # Normalize orth step
            noise_orth = noise_orth / np.linalg.norm(noise_orth)

            # Move along dist sphere
            orth_candidate = x_k + noise_orth * delta * dist

            # Re-project onto the sphere centered at x_original
            orth_direction = orth_candidate - self.x
            orth_direction = orth_direction / np.linalg.norm(orth_direction)
            orth_candidate = self.x + dist * orth_direction

            # Step 3. Nudge the adversary a small distance (epsilon) towards x.
            x_candidate = orth_candidate + epsilon * (self.x - orth_candidate)

            # Clip candidate to ensure it's in the output range
            x_candidate = np.clip(x_candidate, 0.0, 1.0)

            is_adversarial = not self._classify(x_candidate)

            # Accept or Reject candidate
            if is_adversarial:
                x_k = x_candidate
                successes += 1

            # Adjust step sizes accordingly to local geometry
            success_prop = successes / (step + 1)

            # Only tune hyperparameters every 100 iterations
            if success_prop < prop_thres_low:
                # Decrease step sizes, we stepped across the boundary into a non-adversarial zone
                delta = delta * 0.90
                epsilon = epsilon * 0.90
            elif success_prop >= prop_thres_high:
                # Try to accelerate progress, but don't allow delta/epsilon to get too large
                if delta < 0.5:
                    delta = delta * 1.1
                    epsilon = epsilon * 1.1

        # Save images if any are stored
        if xk_points:
            save_points(xk_points, "spook-step")
        return x_k, steps_taken
    