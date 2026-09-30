import sys
sys.path.insert(0, "../")

import numpy as np
from Modules.ANN import Network
from Modules.black_box_attack import BlackBoxAdversary
from Modules.data_io import read_dataset, save_points

# Read in data
train_set, valid_set, test_set = read_dataset()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

in_dim = train_x.shape[1]

# Train new model, print training set accuracy
net = Network(
    [in_dim, 30, 10],
    ["linear", "sigmoid", "sigmoid"],
    loss_function="mse",
    output_mode="one_hot",
)

y_train_enc = net.encode_labels(train_y)
y_val_enc = net.encode_labels(valid_y)

net.SGD(
    train_x, y_train_enc,
    mini_batch_size=32,
    learning_rate=1.0,
    X_val=valid_x, Y_val=y_val_enc,
    max_epoch=10, verbose=False,
)

print("Train accuracy:", net.evaluate(train_x, train_y))
print("Test accuracy:", net.evaluate(test_x, test_y))

################################################################################################

def gen_adv_imgs(
    n_adv: int = 100,
    num_steps: int = 5_000,
    max_adv_dist: float = 5.0,
    starting_idx: int = -1,
):
    """
    Generate new images using the Black Box boundary walk method. Regularly prints out an avg
    dist generated for a new batch of (50) images.

    :param n_adv: The amount of new images to generate.
    :param num_steps: The maximum number of steps along the "walk" for each attack.
    :param max_adv_dist: The maximum allowed distance in L2 norm for a new adversarial image.
    :param starting_idx: The index in the training data to start at. Use -1 to start at the first 
    element.

    :returns tuple:
        Returns the following in order: x_ref, adv_x, adv_y, adv_dist, failed_attempts
        - x_ref: Original image points.
        - adv_x: Newly generated adversarial image points.
        - adv_y: The original correct label for the adversary image.
        - adv_dist: The distance from the original image in L2 norm.
        - failed_attempts: The number of attempts to create an adversarial image that never got 
        below the maximum allowed distance.
    """

    failed_attempts = 0
    x_ref, adv_x, adv_y, adv_dist = [], [], [], []

    # Generate new adversarial images using the black-box decision boundary attack
    i = starting_idx
    while len(adv_x) < n_adv:
        i = (i + 1) % train_x.shape[0]

        x, y = None, None
        if np.argmax(net.predict(train_x[i])) == np.argmax(y_train_enc[i]):
            x = train_x[i]
            y = y_train_enc[i]
        else:
            continue

        # x, y contains a "correctly classified" image point
        adversary = BlackBoxAdversary(net=net, x=x, y=y)

        # Generate a new spook, which must be within a specific distance to x
        spook = None
        for _ in range(5):
            spook = adversary.generate(max_steps=num_steps, verbose=False)
            dist = np.linalg.norm(x - spook)
            
            if dist > max_adv_dist:
                failed_attempts += 1
                continue
            else:
                adv_dist.append(dist)
                adv_x.append(spook)
                adv_y.append(y)
                x_ref.append(x)

                if len(adv_x) % 50 == 0:
                    print(f"Appended adv #{len(adv_x)} -- Dist: {round(np.mean(adv_dist[-50:]), 3)}")
                break

    return x_ref, adv_x, adv_y, adv_dist, failed_attempts

################################################################################################

# Generate new images
print("\n--- Generating Adversarial Images ---")

x_ref, adv_x, adv_y, adv_dist, failed_attempts = gen_adv_imgs(
    n_adv=500,
    num_steps=5000,
    max_adv_dist=10.0,
)

# Verify accuracy is 0%
print(f"\nAdversarial accuracy: {net.evaluate(adv_x, adv_y)}")
print(f"Failed adv generation attempts: {failed_attempts}")

adv_dist = np.array(adv_dist)
print(f"Mean - {round(np.mean(adv_dist), 4)} | StD - {round(np.std(adv_dist), 4)}")
print(f"Largest - {round(np.max(adv_dist), 4)} | Smallest - {round(np.min(adv_dist), 4)}")

# Uncomment below to save some of the spook/original images to compare
# save_points(adv_x[10:25], "spook")
# save_points(x_ref[10:25], "orig")

print("\nRetraining model on new adversarial images...")

# Re-train model on spook data by appending it to the training set
net.SGD(
    np.concat((train_x[:2000], adv_x)), np.concat((y_train_enc[:2000], adv_y)),
    mini_batch_size=32,
    learning_rate=1.0,
    X_val=valid_x, Y_val=y_val_enc,
    max_epoch=10, verbose=False,
)

# Verify that the new accuracy on the spook data is > 90%
print(f"\nAdversarial accuracy: {net.evaluate(adv_x, adv_y)}")
print(f"Test accuracy: {net.evaluate(test_x, test_y)}")

print("\nRegenerating adversarial images")

x_ref, adv_x, adv_y, adv_dist, failed_attempts = gen_adv_imgs(
    n_adv=500,
    num_steps=5000,
    max_adv_dist=10.0,
)

# Verify accuracy is 0%
print(f"\nAdversarial accuracy: {net.evaluate(adv_x, adv_y)}")
print(f"Failed adv generation attempts: {failed_attempts}")

adv_dist = np.array(adv_dist)
print(f"Mean - {round(np.mean(adv_dist), 4)} | StD - {round(np.std(adv_dist), 4)}")
print(f"Largest - {round(np.max(adv_dist), 4)} | Smallest - {round(np.min(adv_dist), 4)}")
