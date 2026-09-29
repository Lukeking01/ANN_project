import gzip
import pickle
import numpy as np
from ANN import Network
from black_box_attack import BlackBoxAdversary
from save_points import save_points

np.random.seed(11)

# Read in data
with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

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
    max_epoch=10,
)

print("Train accuracy:", net.evaluate(train_x, train_y))
print("Test accuracy:", net.evaluate(test_x, test_y))

print("\n--- Generated Adversarial Images ---")
n_adv = 10          # Count of new adversarial images to add
max_adv_dist = 8    # Max dist in 2-norm
failed_attempts = 0

adv_x = []
adv_y = []

# Generate new adversarial images using the black-box decision boundary attack
i = 0
while len(adv_x) < n_adv:
    i += 1

    x, y = None, None
    if np.argmax(net.predict(train_x[i])) == np.argmax(train_y[i]):
        x = train_x[i]
        y = train_y[i]
    else:
        continue

    # x, y contains a "correctly classified" image point
    adversary = BlackBoxAdversary(net=net, x=x, y=y)

    # Generate a new spook, which must be within a specific distance to x
    spook = None
    for _ in range(5):
        spook = adversary.generate(max_steps=25_000, verbose=False)
        dist = np.linalg.norm(x - spook)
        
        if dist > max_adv_dist:
            failed_attempts += 1
            continue
        else:
            adv_x.append(spook)
            adv_y.append(y)
            print(f"Appended adv #{len(adv_x)} -- Dist: {round(dist, 3)}")
            break

print(f"Adversarial accuracy: {net.evaluate(adv_x, adv_y)}")
print(f"Failed adv generation attempts: {failed_attempts}")
save_points(adv_x, f"spook")


# Verify accuracy on spook data is 0%
# TODO

# Re-train model on spook data by appending it to the training set
# TODO

# Verify that the new accuracy on the spook data is > 90%
# TODO

####



# def train_small_model():
#     # Create and train a small neural net on a sample of training data. Returns the
#     # network, and the training data it used.
#     layers = [DenseLayer(
#         train_data[0][0].shape[0], 50), ASigmoid(), 
#         DenseLayer(50, 10), ASigmoid(),
#     ]
#     loss = LQuadratic()
#     net = DenseNet(layers=layers, loss=loss)

#     n = 1000
#     indices = np.random.permutation(train_data[0].shape[0])
#     train_X, train_Y = train_data[0][indices[:n]], train_data[1][indices[:n]]
#     val_X, val_Y = val_data[0][:4000], val_data[1][:4000]

#     net.train(
#         train_X=train_X, train_Y=train_Y,
#         val_X=val_X, val_Y=val_Y,
#         epochs=100, batch_size=200,
#     )

#     print("Neural net fully trained")
#     return net, train_X, train_Y

def test_black_box_gen():
    """
    Test that the black box adversarial generation can successfully generate a spook image
    which gets incorrectly classified, starting from a given correctly-classified image.
    """
    # Mocks
    def train_small_model():
        pass
    def save_points():
        pass
    
    net, train_X, train_Y = train_small_model()
    
    # Find a correctly classified image
    x, y = None, None
    for i in range(100):
        if np.argmax(net.predict(train_X[i])) == np.argmax(train_Y[i]):
            x = train_X[i]
            y = train_Y[i]
            break

    save_points([x], "black-box-target")

    adversary = BlackBoxAdversary(net=net, x=x, y=y)
    spook = adversary.generate()

    y_true = net.predict(x)
    y_spook = net.predict(spook)
    assert np.argmax(y_true) == np.argmax(y)
    assert np.argmax(y_spook) != np.argmax(y)
    save_points([spook], "generated-spook")
    print(f"Classifies target as {np.argmax(y_true)}")
    print(f"Classifies spook as {np.argmax(y_spook)}")

def test_black_box_gen_new_set():
    """
    Test that we can generate a new set of data all visually close to the correctly classified
    images. However, at the end we should have 50 new images all within a distance of at most 1
    from the original images, and the trained network should misclassify all of them (accuracy 0).
    """
    # Stubs
    def train_small_model():
        pass
    
    net, train_X, train_Y = train_small_model()

    new_set_size = 50
    spook_set_X = []
    spook_set_Y = []
    max_spook_dist = 2 # Don't accept spooks further in L-2 norm from a valid datapoint.
    failed_spooks = 0
    
    i = 0
    while len(spook_set_X) < new_set_size:
        i += 1
        
        x, y = None, None
        if np.argmax(net.predict(train_X[i])) == np.argmax(train_Y[i]):
            x = train_X[i]
            y = train_Y[i]
        else:
            continue

        # x, y contains a "correctly classified" image point
        adversary = BlackBoxAdversary(net=net, x=x, y=y)

        # generate a new spook, which must be within a specific distance to x
        spook = None
        for _ in range(5):
            spook = adversary.generate(max_steps=10_000)
            if np.linalg.norm(x - spook) > max_spook_dist:
                failed_spooks += 1
                continue
            else:
                spook_set_X.append(spook)
                spook_set_Y.append(y)
                print(f"Appended spook #{len(spook_set_X)}")
                break
    
    spook_set_X = np.array(spook_set_X)
    spook_set_Y = np.array(spook_set_Y)

    # Model should fail on all images
    assert net.get_accuracy(spook_set_X, spook_set_Y) == 0

    # Uncomment below to save the spook data as images
    # save_points(spook_set_X, "spook")

    # Train model on the generated images, use old training data as validation set
    net.train(
        train_X=spook_set_X, train_Y=spook_set_Y,
        val_X=train_X, val_Y=train_Y,
        epochs=50, batch_size=20,
    )

    assert net.get_accuracy(spook_set_X, spook_set_Y) > 80
