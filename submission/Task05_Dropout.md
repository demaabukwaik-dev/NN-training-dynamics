# Task 05 — Dropout Ablation Study

## 1. Objective

Our goal is to analyze how Dropout affects the model's training behavior and its ability to generalize, we will train three identical models with different Dropout rates (0, 0.1, and 0.3) and compare their overfitting levels to understand how Dropout prevents neuron co-adaptation.


## 2. Code Used

```python
# Train the model with different dropouts

histories = [] # storing the histories of the models with different dropouts in a list to plot later
dropouts_list = [0, 0.1, 0.3] # the dropouts we want to train the model with

# Train the model with different dropouts
for d in dropouts_list:
    print(f"\n--- Training with  {d} dropout ---")

    # build model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(d),
        keras.layers.Dense(10, activation="softmax")
    ]) 

    # compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # train model
    history = model.fit(x_tr, y_tr,epochs=20,batch_size=32,validation_data=(x_val, y_val))

    # store history of the model
    histories.append(history)
```


## 3. Results
- Dropout = 0:   Train Loss: 0.0056 | Val Loss: 0.1115 (Severe Overfitting) — the generalization gap is large
- Dropout = 0.1: Train Loss: 0.0173 | Val Loss: 0.0912 (Moderate Overfitting) — the gap becomes smaller
- Dropout = 0.3: Train Loss: 0.0419 | Val Loss: 0.0841 (Low Overfitting) — the gap becomes much smaller

_(Plots showing the loss curves for all runs are saved in `results/loss_curves/`)_


## 4. Short Analysis

### 1. Is there overfitting?

To determine whether overfitting occurred, we have to look at the final train loss and validation loss for each Dropout and compare the gaps between them. The larger the gap between them, the more the model has memorized the training data instead of learning general patterns (overfitting).

| **Dropout Rate** | **Final Train Loss** | **Final Val Loss** | **Generalization Gap** | **Status**           |
| ---------------- | -------------------- | ------------------ | ---------------------- | -------------------- |
| 0.0              | 0.0056               | 0.1115             | 0.1059                 | Severe Overfitting   |
| 0.1              | 0.0173               | 0.0912             | 0.0739                 | Moderate Overfitting |
| 0.3              | 0.0419               | 0.0841             | 0.0422                 | Generalizing Well    |

---

### 2. When did the model start to overfit?

To know this we have to track the learning process by observing the val_loss behavior from the points where train and validation loss intersect.

**Dropout = 0:** The val_loss stopped decreasing at around epoch 5, and started oscillating very sharply and increasing, going far from the train loss and forming a large gap (since the train loss continues decreasing toward 0, meaning the model is memorizing the training data) so it started severe overfitting at around epoch 5.

**Dropout = 0.1:** The val_loss stopped decreasing at around epoch 7 (a bit later). After that, it also started oscillating but with less intensity and increasing, forming a gap with the train loss but the gap formed here is much smaller than the gap with no dropout so the model started overfitting at around epoch 7 (start of overfitting is delayed and the intensity is much lower).

**Dropout = 0.3:** The val_loss stopped decreasing at around epoch 10 (even later). After that, it also started oscillating slightly and increasing, forming a gap with the train loss. However, the gap formed here is much smaller than the gap with 0.1 so the model started overfitting at around epoch 10 (start of overfitting is further delayed and the intensity is much lower).

**Note on Training Speed:** We can also notice how Dropout affected the speed of learning, with Dropout = 0 the train loss decreased very fast. With Dropout = 0.1, it decreased slower, with Dropout = 0.3 the train loss decreased much slower. This happens because turning off neurons makes the learning process harder and slower for the model.

The co-adaptation means that during the training process, neurons learn to depend on each other instead of learning the actual patterns, dropout prevents this by deactivating some neurons randomly at each step, so the neuron is forced to depend on itself  because it will not be able to know whether the other neuron next to it will be on or off, making it doing better in generalizing on unseen data. This is what we call robust representations.

## 5. Key Takeaway

There is a trade-off when using Dropout, increasing Dropout makes the learning process slower, but it reduces overfitting by forcing the model to build robust representations.