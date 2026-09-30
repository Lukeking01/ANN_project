import sys
sys.path.insert(0, "../")

import numpy as np
from Modules.ANN import Network
from Modules.data_io import read_dataset

np.random.seed(0)

train_set, _, _ = read_dataset()


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