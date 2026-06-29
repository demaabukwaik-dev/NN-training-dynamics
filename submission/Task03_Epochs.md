# Task 03 — Epoch-Based Learning Curve Exploration

## 1. Objective

Our goal is to train the model with 5, 10, and 20 epochs to compare training dynamics, identify signs of overfitting, and explain how the Adam optimizer influenced the speed and stability of convergence.


## 2. Code Used

```python
histories = []
epoch_list = [5, 10, 20]
for e in epoch_list:
    print(f"\n--- Training with max {e} epochs ---")
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) 

    history = model.fit(x_tr, y_tr, epochs=e, batch_size=32, validation_data=(x_val, y_val))
    
    histories.append(history)
```

## 3. Results

- 5 Epochs: Train Loss: 0.0627 | Val Loss: 0.0788 | Val Accuracy: 0.9766 (No overfitting)
- 10 Epochs: Train Loss: 0.0308 | Val Loss: 0.0818 | Val Accuracy: 0.9796 (Overfitting started)
- 20 Epochs: Train Loss: 0.0166 | Val Loss: 0.1001 | Val Accuracy: 0.9788 (Severe overfitting)

_(Plots showing the loss and accuracy curves for all runs are saved in `results/loss_curves/`)_


## 4. Short Analysis

As mentioned, our goal is to analyze why the model's behavior changed as we increased the epochs. Here is how the training dynamics evolved:

### Training Dynamics & Overfitting Signs

**5 Epochs:**

1. **Visual Observation**:

- Both loss lines decrease, intersect at Epoch 4, and descend together with almost no gap.

- Val_loss decreased until Epoch 5 (0.0788) following the normal ternd Since the val_loss continuews dereasing this is not overfitting and the model is  generalizing well.

2. **Generalization Gap**:

This is supported by the loss gap between Train loss (0.0627) and Val loss (0.0799), which is almost zero. Similarly, the gap between Train accuracy (0.9814) and Val accuracy (0.9788) is almost zero, confirming a perfectly balanced model with no overfitting.

**10 Epochs:**

1. **Visual Observation**:

- Both loss lines intersect at Epoch 4. After that, train loss keeps decreasing while val loss oscillates (a little) and finally increases, forming a gap.

- Val_loss decreased and reached a low point at Epoch 4 (0.0797) and increased slightly at at Epoch 5 and 6, then from epoch 7 t0 10 the model was oscillating slightly and finally increasing reaching 0.0818, train loss kept decreasing while val_loss increased  Here the increase is not only noise, it is a real trend, meaning the model started memorizing the training data(overfitting).

2. **Generalization Gap**:

This is supported by the loss gap getting wider between Train loss (0.0308) and Val loss (0.0818). Also, the gap between Train accuracy (0.9898) and Val accuracy (0.9796) became 1%, confirming the model performs a little better on training data than unseen data (this is exactly overfitting).

**20 Epochs:**

1. **Visual Observation**:

- Both loss lines intersect at around Epoch 4. After that, train loss keeps decreasing a lot while val loss oscillates a lot between decreasing and increasing, forming a wide gap.

- Val_loss decreased until it reached its best point at Epoch 5 (0.0761). After that and through the rest of the epochs, it oscillates a lot between increasing and decreasing, the highest val_loss reached was 0.1001 at Epoch 20. Here it is very clear that the increasing is a real trend, meaning the model not only started memorizing the training data but it actually memorized it.

2. **Generalization Gap**:

This is supported by the large loss gap getting wider and wider between Train loss (0.0166) and Val loss (0.1001). Also, the gap between Train accuracy (0.9945) and Val accuracy (0.9788) became 1%+, confirming the model performs much better on training data than unseen data (this is severe overfitting).

### How the Optimizer (Adam) Influenced Convergence

Adam dynamically adapts the learning rate for each parameter. This directly explains the behavior we saw across the three runs:

First, Adam is responsible for the high-speed convergence seen in all three runs. foe example while training with 20 epochs in the first few epochs (specifically Epoch 2) the train loss dropped sharply (from 0.29 to 0.13), showing Adam adjusted weights in a good way and took large steps initially to learn the patterns fast.

However because Adam is so effective at minimizing the training loss (forcing), it becomes the reason for overfitting if training is not stopped. In the 5 epochs run, 5 epochs were enough because Adam found good weights quickly. But in the 10 epochs run, 10 epochs were actually too many because Adam already found the optimal weights at Epoch 6, and when we continued to train, Adam optimized the weights too much on the training set, leading to overfitting. This got worse in the 20 epochs run, where Adam found the optimal weights at Epoch 5, and forcing it to continue for 15 more epochs caused it to memorize the training data, leading to severe overfitting.

## 5. Key Takeaway

More epochs don't always mean better performance since Adam finds the optimal weights quickly, and continuing to train after that point only causes the model to memorize the training data and lose its ability to generalize (overfitting).