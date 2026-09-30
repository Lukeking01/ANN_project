import gzip
import pickle
import numpy as np
from ANN import Network

np.random.seed(0)

with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_x, train_y = train_set

X50 = train_x[:50]
Y50 = np.eye(10)[train_y[:50]]
eta=0.1

net = Network(
    structure=[784, 10, 10],
    activation_functions=["linear", "linear", "sigmoid"],
    loss_function="mse",
    output_mode="one_hot",
)

_, _, final_loss, epochs=net.SGD(
    X50, Y50,
    mini_batch_size=5,     
    learning_rate=eta,     
    max_epoch=100000,      
    tolerance=1e-8,
    step_tol=0.0,         
    verbose=False,
    learning_rate_decay=1.001
)

loss = net.calculate_loss(X50, Y50)
print(f"epochs: {epochs}")
print(f"Final loss: {loss:.3e}")