import gzip
import pickle
import numpy as np
from ANN import Network

np.random.seed(11)

with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

net = Network(
    [784, 128, 64, 10],
    ["linear","sigmoid", "sigmoid"],
    loss_function="mse",
    output_mode="one_hot",
)

Y_train_enc = net.encode_labels(train_y)
Y_val_enc = net.encode_labels(valid_y)

net.SGD(
    train_x, Y_train_enc, mini_batch_size=32, learning_rate=0.5,
    X_val=valid_x, Y_val=Y_val_enc,max_epoch=10,
)
print("Test accuracy:", net.evaluate(test_x, test_y))


net.confusion_matrix(test_x, test_y, nr_labels=10)