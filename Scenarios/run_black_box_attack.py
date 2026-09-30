import sys
sys.path.insert(0, "../")

import gzip
import pickle
import numpy as np
from ANN import Network
from black_box_attack import BlackBoxAdversary
from save_points import save_points

# Read in data
with gzip.open("../data/mnist.pkl.gz", "rb") as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

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

print("\n--- Generated Adversarial Images ---")
num_steps = 5_000
n_adv = 500              # Count of new adversarial images to add
max_adv_dist = 10.0      # Max dist in 2-norm
failed_attempts = 0

x_ref = []
adv_x = []
adv_y = []
adv_dist = []

# Generate new adversarial images using the black-box decision boundary attack
i = -1
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
            print(f"Appended adv #{len(adv_x)} -- Dist: {round(dist, 3)}")
            break

# Verify accuracy is 0%
print(f"Adversarial accuracy: {net.evaluate(adv_x, adv_y)}")
print(f"Failed adv generation attempts: {failed_attempts}")

adv_dist = np.array(adv_dist)
print(f"Mean - {round(np.mean(adv_dist), 4)} | StD - {round(np.std(adv_dist), 4)}")
print(f"Largest - {round(np.max(adv_dist), 4)} | Smallest - {round(np.min(adv_dist), 4)}")

# Uncomment below to save some of the spook/original images to compare
# save_points(adv_x[10:25], "spook")
# save_points(x_ref[10:25], "orig")

# Re-train model on spook data by appending it to the training set
net.SGD(
    np.concat((train_x[:2000], adv_x)), np.concat((y_train_enc[:2000], adv_y)),
    mini_batch_size=32,
    learning_rate=1.0,
    X_val=valid_x, Y_val=y_val_enc,
    max_epoch=10, verbose=False,
)

# Verify that the new accuracy on the spook data is > 90%
print(f"Adversarial accuracy: {net.evaluate(adv_x, adv_y)}")
print(f"Test accuracy: {net.evaluate(test_x, test_y)}")

# TODO Extract generation logic into a reusable function
# TODO Add verbosity flag so that we don't print n_adv lines to the console every time.
print("Regenerating adversarial images")

failed_attempts = 0

x_ref = []
adv_x = []
adv_y = []
adv_dist = []

# Generate new adversarial images using the black-box decision boundary attack
i = -1
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
            print(f"Appended adv #{len(adv_x)} -- Dist: {round(dist, 3)}")
            break

# Verify accuracy is 0%
print(f"Adversarial accuracy: {net.evaluate(adv_x, adv_y)}")
print(f"Failed adv generation attempts: {failed_attempts}")

adv_dist = np.array(adv_dist)
print(f"Mean - {round(np.mean(adv_dist), 4)} | StD - {round(np.std(adv_dist), 4)}")
print(f"Largest - {round(np.max(adv_dist), 4)} | Smallest - {round(np.min(adv_dist), 4)}")
