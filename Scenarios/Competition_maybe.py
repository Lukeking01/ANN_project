
import sys
sys.path.insert(0, "../")

import gzip
import pickle
import numpy as np
import ANN
import matplotlib.pyplot as plt

with gzip.open("../data/mnist.pkl.gz", "rb") as file:
    data = pickle.load(file, encoding="latin1")

training_data, validation_data, test_data = data

training_inputs, training_labels = training_data
validation_inputs, validation_labels = validation_data
test_inputs, test_labels = test_data

X_train = training_inputs
Y_train = np.array([ANN.one_hot(y) for y in training_labels])

X_val = validation_inputs
Y_val = np.array([ANN.one_hot(y) for y in validation_labels])

X_test = test_inputs
Y_test = np.array([ANN.one_hot(y) for y in test_labels])

network_structure = [784,3,  10]
activation_functions = ["linear","linear", "step"]
activation_derivative = [ANN.sigmoid_derivative,lambda z:1,ANN.sigmoid_derivative]

for _ in range(15):
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
        max_epoch = 800,
        step_tol = 0,
        verbose = False
    )
    
    print(f"Reached loss: {loss}, in {epoch} epochs.")



overtrain_network.confusion_matrix(X_train[:cutoff], Y_train[:cutoff], 10)
