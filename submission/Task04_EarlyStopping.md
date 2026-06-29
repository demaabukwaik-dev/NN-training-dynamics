# Task 04 — EarlyStopping Behavior Analysis

## 1. Objective

Our goal is to analyze how EarlyStopping monitors the training process, we will see the patience and how it affects the time the training stops, why we use validation loss as the metric that controls this decision, what happens when we increase patience or when we change the optimizer, and how EarlyStopping acts as an indirect form of regularization.


## 2. Code Used

```python
# try the model with different patiences (3 vs 5)

histories = [] # storing the histories of the models with different patiences in a list to plot later
paitence_list = [3,5] # the patience values we want to train the model with

# Train the model with different patiences
for p in paitence_list:
    print(f"\n--- Training with max 20 epochs ---")
    print(f"\n--- Training with {p} patience ---")

    # build model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(10, activation="softmax")
    ]) 

    # compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # early stopping
    callbacks = [
    keras.callbacks.EarlyStopping(patience=p, restore_best_weights=True)
    ]

    # train model
    history = model.fit(x_tr, y_tr,epochs=20,batch_size=32,validation_data=(x_val, y_val), callbacks=callbacks)

    # store history of the model
    histories.append(history)
```



## 3. Results
- p = 3: Stopped at Epoch 11  | Best Val Loss: 0.0689 (at Epoch 8)
- p = 5: Stopped at Epoch 12 | Best Val Loss: 0.0709 (at Epoch 7)
- SGD Optimizer: Did not stop (completed 20 epochs) | Final Val Loss: 0.0992

_(Plots showing the loss curves for all runs are saved in `results/loss_curves/`)_


## 4. Short Analysis

#### 1. At which epoch did training stop?

**p = 3:**

- Val_loss decreased until it reached its best point at Epoch 8 (0.0689) → saved as best weights.
- Continue:
  - At Epoch 9 val_loss increases  → count = 1
  - At Epoch 10 val_loss increases → count = 2
  - At Epoch 11 val_loss decreases but > best → count = 3
  - The 3 trials were consumed so the training stopped at Epoch 11.
  - The best saved weights from Epoch 8 are considered.

**p = 5:**

- Val_loss decreased until it reached its best point at Epoch 3 (0.0843) → saved as best weights.
- Continue:
  - At Epoch 4 val_loss increases  → count = 1
  - At Epoch 5 val_loss decreases < best   → (new best is updated (0.0800) and counter resets)
  - Continue:
    - At Epoch 6 val_loss decreases < best → (new best is updated (0.0731) and counter resets)
    - At Epoch 7 val_loss decreases < best → (new best is updated (0.0709) and counter resets)
    - At Epoch 8 val_loss increases  → count = 1
    - At Epoch 9 val_loss increases  → count = 2
    - At Epoch 10 val_loss increases → count = 3
    - At Epoch 11 val_loss increases → count = 4
    - At Epoch 12 val_loss decreases but > best → count = 5
    - The 5 trials were consumed so the training stopped at Epoch 12.
    - The best saved weights from Epoch 7 are considered.


#### 2. Why does the validation loss control this decision?

The normal behavior of training loss is to continuously decrease as training continues so it is not a reliable metric to detect overfitting, since it may keep improving even when the model starts memorizing the training data.

In contrast, validation loss measures performance on unseen data, making it a more reliable indicator of generalization because when validation loss starts increasing, it indicates that the model is no longer learning general patterns but instead beginning to overfit.

this is why EarlyStopping relies on validation loss rather than training loss.

#### 3. What happens if you increase patience (e.g., to 5)?

Increasing patience allows the model to continue training for more epochs even after validation loss stops improving. This can sometimes help the model reach a better minimum, as seen in the experiment:

- p = 3 → stops at Epoch 11 with val_loss = 0.0689
- p = 5 → stops at Epoch 12 with val_loss = 0.0709 

increasing patience here allowed the model to continue training during periods where validation loss is already getting worse, which can lead to overfitting.

#### 4. Would a different optimizer (e.g., SGD) change the EarlyStopping pattern?

Yes the optimizer affects convergence speed.

- Adam converges quickly, so EarlyStopping is triggered earlier (around Epoch 11).
- SGD converges more slowly, so validation loss improves gradually, and no EarlyStopping occurred within 20 epochs.

Therefore, the optimizer directly affects when EarlyStopping is activated.

#### 5. Explain how EarlyStopping acts as an indirect form of regularization.

EarlyStopping acts as a form of regularization by limiting training time rather than modifying the model itself since othre regularization methods like Dropout directly modify how the model learns but EarlyStopping prevents overfitting by stopping training at the point where validation loss begins to increase.

It is considered an indirect form of regularization because it controls when learning stops, not how the model is built.

## 5. Key Takeaway

EarlyStopping prevents overfitting not by changing the model architecture, but by stopping training at the exact moment the model stops learning general patterns and starts overfitting.