# Task 01 — Prediction Analysis

## 1. Objective

Our goal is to run the trained model on three samples from `x_test`, compare the predicted labels with the true labels and then provide an explanation to tell why the model predicted correctly by tracing the flow of input data through network layeers and how Adam optimizer shaped the weights during training.

## 2. Code Used

```python
samples = [sample1 , sample2 , sample3]
  
for i, sample in enumerate(samples, start=13):
    pred = model.predict(sample)
    print("Predicted Label:", np.argmax(pred))
    print("Confidence:", np.max(pred)*100)  
    print("True Label:", y_test[i])
```

## 3. Results

- Sample 13 - Predicted: 0 (99.99%) | True: 0
- Sample 14 - Predicted: 1 (99.99%) | True: 1
- Sample 15 - Predicted: 5 (99.68%) | True: 5

Note that the model predicted the digits with a confidence above 99%, indicating the model is very certain of its prediction.

_(Images showing the digits and their predictions are saved in `results/predictions/`)_


## 4. Short Analysis

here we will understand why the model predicted these three samples correctly by tracing how the data moved through the network:

### 1) Input Layer

- Each image is 28×28 pixels = 784 values.
- The image is flattened from a 2D matrix to a 1D vector (784).
- The network receives 784 normalized values (between 0 and 1).

### 2) Hidden Layer (Dense 128)

- The neural network consists of only 1 dense layer consisting of 128 neurons.
- Each neuron receives the 784 inputs and computes a weighted sum: `z = (x1*w1 + x2*w2 + ... + x784*w784) + bias` (where w1, w2, ... are the learned weights).

- Then, for each z, the **ReLU** activation is applied to make the model understand non-linear relations not only linear ones. It removes negative values and keeps positive values, which helps the model focus on inputs that are really useful in prediction.

- **Dropout(0.1)** means 10% of neurons are deactivated randomly during training to reduce overfitting.

### 3) Output Layer

- The output layer consists of 10 neurons (classes).
- Each neuron computes: `z = (a1*w1 + a2*w2 + ... + a128*w128) + bias` (where a1, a2, ... are the 128 outputs from the hidden layer).

- The **Softmax** activation function is then applied on z which converts the outputs into probabilities for each digit (0–9) that sum up to 1. The class with the highest probability is chosen to be the predicted value using `argmax`.

### How the optimizer (Adam) shaped weight updates during training:

If the loss resulting from the forward pass is high then we need to do backpropagation to reduce the loss and during training the Adam optimizer updated the weights iteratively until it reached the optimal weights:

First, the model was set to be trained on the entire training data (55,000 samples) a maximum of 20 times (`epochs=20`), but due to `EarlyStopping(patience=3)` the training was stopped early to prevent overfitting.

Adam does not update the weights of all 55,000 images at once (which is slow) but it updates the weights using batches each consisting of 32 images, meaning that in **every epoch** the weights were updated 55,000 ÷ 32 = **1,718 times**.

In each of these 1,718 steps, Adam calculated the **Training Loss** for each batch of 32 images, calculated the gradients, and updated the weights to reduce this error. (note: the "training loss" value that appears at the end of an epoch, or that we plotted on graphs is not used for updates, it is the **average** of all 1,718 batch losses for the current epoch, used only for monitoring.)

At the end of each epoch, we have the **validation loss** which is calculated on the validation set and used by `EarlyStopping` for monitoring. Once it noticed the validation loss stopped improving for 3 epochs, it stopped the training.

**Tracking val_loss to see how EarlyStopping acts:**

20 epochs were specified, but the model stopped at epoch 9 due to early stopping:
- **Epoch 6:** val_loss reached its lowest point at `0.0690` (Best Weights were saved here).
- **Epoch 7:** val_loss increased to `0.0725` (Patience counter = 1).
- **Epoch 8:** val_loss decreased to `0.0694` but still >  `0.0690`(Patience counter = 2).
- **Epoch 9:** val_loss increased again to `0.0748` (Patience counter = 3 → **Training Stopped**).

the model predicted (0, 1, and 5) correctly means Adam succeeded in finding good weights describing the patterns, so we can conclude that the architecture of our neural network was well-structured and captured the non-linear relation so it gave correct results on unseen data.


## 5. Key Takeaway

A correct prediction is an expected result of a well-structured forward pass (Flatten → ReLU → Softmax) with an optimizer (Adam) that carefully updated weights in batches, with EarlyStopping ensuring those weights remained generalized to unseen data.