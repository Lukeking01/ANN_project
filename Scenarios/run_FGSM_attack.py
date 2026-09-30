import sys
sys.path.insert(0, "../")

from Modules.ANN import Network
from Modules.FGSM_attack import FGSMAdversary
from Modules.data_io import read_dataset

train_set, valid_set, test_set = read_dataset()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

epsilon = 0.15

print("Training Base Model")
net = Network(
    [784, 30, 10],
    ["linear", "sigmoid", "sigmoid"],
    loss_function="mse",
)
Y_train_enc = net.encode_labels(train_y)
Y_val_enc = net.encode_labels(valid_y)
net.SGD(
    train_x, Y_train_enc,
    mini_batch_size=32,
    learning_rate=0.5,
    X_val=valid_x, Y_val=Y_val_enc,
    max_epoch=5,
)

print("Generating FGSM Adversarial Examples")
atk = FGSMAdversary(net, valid_x, Y_val_enc)
valid_x_adv = atk.generate(eps=epsilon, verbose=True)

acc_clean = net.evaluate(valid_x, valid_y)
acc_adv = net.evaluate(valid_x_adv, valid_y)

print(f"Model Accuracy on Clean Data: {acc_clean * 100:.2f}%")
print(f"Model Accuracy on FGSM Data:  {acc_adv * 100:.2f}%")