# ANN_project
Project 2 for course NUMN21.

## Files and Usage Notes

- **Modules:** Implementation Code
    - `ANN.py`: Neural network class, loss, and activation functions.
    - `black_box_attack.py`: Adversarial attack using steps along the decision boundary.
    - `FGSM_attack.py`: Adversarial attack using the sign of the gradient of the model's loss with respect to the input.
    - `data_io.py`: Useful functions for reading in data, and saving data images.
- **Scenarios:** Scripts running various tasks on the neural net (ANN) class.
    - `run_bitwise.py`: Trains ANN with 4 output neurons (bit-wise encoded).
    - `run_black_box_attack.py`: Attacks trained network with boundary attack, then re-trains on those generated samples.
    - `run_competition.py`: Competition attempt with 10 hidden neurons.
    - `run_competition_maybe.py`: Competition attempt with only 3 hidden neurons.
    - `run_FGSM_attack.py`: Attack the trained network with FGSM, then re-trains on the generated samples.
    - `run_onehot.py`: Trains ANN with 10 output neurons (one-hot encoded).
- **Example_usage.py:** Reference for model parameters and useful features.

---

To run any of the Scenarios, `cd` into the scenarios folder and run the files from there.

## Example of the input images

## Task 6 (Output of the learning success per epoch)
![image](Images/elbow.png)
```
Epoch 1  Training loss: 4.5568e-01  Validation loss: 4.5589e-01  Validation accuracy: 20.87%
Epoch 2  Training loss: 4.0797e-01  Validation loss: 4.0732e-01  Validation accuracy: 44.33%
Epoch 3  Training loss: 3.8779e-01  Validation loss: 3.8680e-01  Validation accuracy: 50.36%
Epoch 4  Training loss: 3.1656e-01  Validation loss: 3.1318e-01  Validation accuracy: 69.35%
Epoch 5  Training loss: 2.6518e-01  Validation loss: 2.5953e-01  Validation accuracy: 78.94%
```

## Task 8 (Challenge best result)

> Reached loss: 0.0, in 85 epochs using 3 hidden nodes.

# Extension

## Task 2 (Comparison of the bitwise representation)
 > After 10 epochs we end up with the training loss of 0.19156, the validation loss of 0.27674 and validation accuracy of 93.17%. The test accuracy is 92.93%. During this process 39 predictions were invalid (predicted larger than zero). The confusion matrix is the following:

Figure_bitwise

## Task 5 (Results after the additional attack)

Ran the decision boundary walk attack to generate 500 new images with a maximum distance from valid, correctly classified images, of 10.0 in the 2-norm. By definition, the accuracy of the model on those images is 0%. After training on the model again, we ran the attack one more time, and the model shows some robustness at resisting the same attack.

```
    --- Starting test accuracy ---
Train accuracy: 0.9599
Test accuracy: 0.9535

    --- First Boundary Adversarial Attack (500 new images) ---
Adversarial accuracy: 0.0
Failed adv generation attempts: 1

Mean - 3.7299 | StD - 1.9247
Largest - 9.9233 | Smallest - 0.1309

   --- After training the model on the new images appended to a large subset of old training data ---
Adversarial accuracy: 1.0
Test accuracy: 0.9465

    --- Second Boundary Adversarial Attack (500 new images) ---
Adversarial accuracy: 0.0
Failed adv generation attempts: 57

Mean - 4.7979 | StD - 2.11
Largest - 9.9644 | Smallest - 0.2175
```

### Example of a poor adversarial image

The generated image had a diff in the 2-norm of ~6.57.

<img src="./Images/bad-spook-6.57.png" alt="drawing" width="100" style="display: inline-block; margin-inline: 2em;">
<img src="./Images/bad-spook-ref.png" alt="drawing" width="100" style="display: inline-block;" />

### Example of a good adversarial image

The generated image had a diff in the 2-norm of ~0.26.

<img src="./Images/good-spook-0.26.png" alt="drawing" width="100" style="display: inline-block; margin-inline: 2em;">
<img src="./Images/good-spook-ref.png" alt="drawing" width="100" style="display: inline-block;" />

## Task 5* (Same experiment, but with a more robust attack implementation)

This attack was after the hyper-parameter tuning of delta/epsilon in the black-box attack method
were improved. Average adversarial image distance was improved greatly, so we lowered the maximum allowed distance to 5.0, and generated 5,000 images twice to compare the attack.

It can be seen that generating new fake images after the model was retrained was less successful
(706 failed attempts compared to 111). And, the mean distance achieved went up to ~2.7 from ~2.1.

```
    --- Starting test accuracy ---
Train accuracy: 0.9574
Test accuracy: 0.9510

    --- First Boundary Adversarial Attack (5000 new images) ---
Adversarial accuracy: 0.0
Failed adv generation attempts: 111

Mean - 2.0534 | StD - 0.9716
Largest - 4.9925 | Smallest - 0.0129

   --- After training the model on the new images appended to a large subset of old training data ---
Adversarial accuracy: 0.9856
Test accuracy: 0.9403

    --- Second Boundary Adversarial Attack (5000 new images) ---
Adversarial accuracy: 0.0
Failed adv generation attempts: 706

Mean - 2.7121 | StD - 1.1078
Largest - 4.9997 | Smallest - 0.0292
```

## Contributions

We have all individually finished the assignment and then decided to combine the best parts of each of our codes.
- Linn Preuss Jelvez - Minimization of neural network size for the overfit competition.
- Lukas Nord - Minimization of neural network size for the overfit competition, using questionable methods. 
- Scott Gibson - Decision Based Boundary Adversarial Attack
- Orsolya Bosáková - Bitwise representation implementation and comparison, FGSM attack

## TO DO:
- Finish the readme
- Update slides for presentation
