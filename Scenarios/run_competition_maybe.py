
import sys
sys.path.insert(0, "../")

import numpy as np
import Modules.ANN as ANN
import matplotlib.pyplot as plt
from Modules.data_io import read_dataset

training_data, validation_data, test_data = read_dataset()

training_inputs, training_labels = training_data
validation_inputs, validation_labels = validation_data
test_inputs, test_labels = test_data

X_train = training_inputs
Y_train = np.array([ANN.one_hot(y) for y in training_labels])

X_val = validation_inputs
Y_val = np.array([ANN.one_hot(y) for y in validation_labels])

X_test = test_inputs
Y_test = np.array([ANN.one_hot(y) for y in test_labels])

network_structure = [784, 2, 10]
activation_functions = ["linear", "linear", "step"]
activation_derivative = [ANN.linear_derivative, ANN.linear_derivative, ANN.sigmoid_derivative]

static = 2

##### Seeds scanned for 1, 2, 3 hidden nodes. Epochs limited at 10000 for seeds in range 100
### 1 hidden node
# seed = ? 
### 2 hidden nodes
# seed = 0, 2, 4, 11, 12, 20, 21, 22, 25, 27, 29, 42, 45, 48, 50, 57, 70, 74, 76, 78, 82, 83, 86, 87, 91, 94, 95, 98
    # totaling 28/100. Best at seed = 21, 524 epochs
### 3 hidden nodes
# seed = not [] totaling 100/100 seeds
    # Best at seed = 74, 52 epochs

# seed = 21
for _ in range(100):
    if static:
        seed = _
        np.random.seed(seed)
    overtrain_network = ANN.Network(network_structure,
                    activation_functions,
                    activation_derivative,
                    loss_function = "mse"
                    )


    cutoff = 50


    val,train,loss,epoch = overtrain_network.SGD(
        X_train=X_train[:cutoff],
        Y_train=Y_train[:cutoff],
        mini_batch_size=5,
        learning_rate=0.05,
        tolerance=1e-10,
        max_epoch = 10000,
        step_tol = 0,
        verbose = False
    )
    
    if loss == 0:
        if static == 2:
            print(f"Reached loss: {loss}, in {epoch} epochs. Seed = {seed}")
        else:
            print(f"Reached loss: {loss}, in {epoch} epochs.")



# overtrain_network.confusion_matrix(X_train[:cutoff], Y_train[:cutoff], 10)
