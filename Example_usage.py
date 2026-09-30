
import gzip
import pickle
import numpy as np
import ANN
import matplotlib.pyplot as plt

## Open the data
with gzip.open("data/mnist.pkl.gz", "rb") as file:
    data = pickle.load(file, encoding="latin1")

training_data, validation_data, test_data = data

training_inputs, training_labels = training_data
validation_inputs, validation_labels = validation_data
test_inputs, test_labels = test_data

## one_hot encoding
X_train = training_inputs
Y_train = np.array([ANN.one_hot(y) for y in training_labels])

X_val = validation_inputs
Y_val = np.array([ANN.one_hot(y) for y in validation_labels])

X_test = test_inputs
Y_test = np.array([ANN.one_hot(y) for y in test_labels])



## Init ANN network
Example = ANN.Network(
    structure = [784, 30, 10], # list of number of nodes in each layer. Any number of layers
        activation_functions= ["linear","sigmoid","sigmoid"], # one activation function per laayer. first is for input and should be kept as "linear"
        activation_derivatives = [ANN.linear_derivative,ANN.sigmoid_derivative,ANN.sigmoid_derivative], # list of derivatives of activation functions. Will default to derivatives of the respective activation functions if not given.
        loss_function="mse", # Choice of loss function. Defaults to MSE
        #loss_derivative = None #  Optional choice of derivative of loss function. Defaults to the derivative of loss_function
        )

Validation_losses, training_losses, final_loss, final_epoch = Example.SGD(
        X_train = X_train, # training inputs
        Y_train = Y_train, # training labels
        mini_batch_size=1000, # size of mini_batch, default 1
        learning_rate=0.1, # learning rate, default 0.1
        X_val=X_val, # validation inputs
        Y_val=Y_val, # validation labels
        tolerance=1e-4, # training stops if training loss is less than this, default 1e-4
        verbose=True, # Set to false to suppress training outputs, default True
        max_epoch=10, # maximum epochs trained for, default 2e4
        step_tol=1e-5, # training stops if an individual epoch does not change the training loss by at least this much, default 1e-5
        learning_rate_decay = "random" # If set, learning rate is multiplied by this number each epoch, default 0 = not active
    )

############## Example of usage of methods in the class
print("\n \n \n")
# predicts singular output
prediction = Example.predict(X_test[1]) 
print(f"The prediction is {np.argmax(prediction)}, and should have been {np.argmax(Y_test[1])}","\n")

# predicts output for a batch
batch_prediction = Example.predict_batch(X_test[23:26]) 
print(f"The prediction for the batch is {[int(np.argmax(pred)) for pred in batch_prediction]} \n \t   and should have been {[int(np.argmax(pred)) for pred in Y_test[23:26]]}","\n")

# calculates the loss for the given data
test_loss = Example.calculate_loss(X_test, Y_test) 
print(f"Test_loss = {test_loss}","\n")

# returns how accurate the models predictions are 0-1
evaluation = Example.evaluate(X_test, Y_test) 
print(f"The evaluation of the model is: {evaluation}","\n")

# plots a confusion matrix for the given data
conf_matrix = Example.confusion_matrix(X = X_test, Y = Y_test, nr_labels = 10, labels = True) 

# saves the entire trained model using the path
Example.save("saved_models/Example_network.npz") 

# loads a saved network
New_network = ANN.Network.load("saved_models/Example_network.npz") 


plt.plot(Validation_losses, label="validation")
plt.plot(training_losses, label="training")
plt.legend()
plt.show()