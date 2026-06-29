# Task 08 — Batch Size & Gradient Noise Experiment

## 1. Objective

Our goal is to analyze how different batch sizes affect the training process, we will train three identical models with batch sizes of 8, 32, and 128, then compare how batch size influences gradient noise, the smoothness of loss curves, convergence speed, and the model's ability to generalize.

## 2. Code Used

```python
# Train the model with different batch sizes

histories = [] # storing the histories of the models with different batch sizes in a list to plot later
batch_size_list = [8, 32, 128] # the batch sizes we want to train the model with

# Train the model with different batch sizes
for b in batch_size_list:
    print(f"\n--- Training with  {b} batch_size ---")

    # build model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(10, activation="softmax")
    ]) 

    # compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    # train model
    history = model.fit(x_tr, y_tr,epochs=20,batch_size=b,validation_data=(x_val, y_val))

    # store history of the model
    histories.append(history)
```


## 3. Results
- Batch Size = 8:   Train Loss: 0.0277 | Val Loss: 0.1199 (Severe Overfitting)
- Batch Size = 32:  Train Loss: 0.0148 | Val Loss: 0.0952 (Moderate Overfitting)
- Batch Size = 128: Train Loss: 0.0201 | Val Loss: 0.0773 (Best Generalization)

_(Plots showing the loss curves for all runs are saved in `results/loss_curves/`)_


## 4. Short Analysis

### How batch size affects the smoothness of loss curves & Why smaller batches introduce gradient noise:

From the plots of train loss, we notice that:
- for Batch Size = 8 the curve oscillates heavily with a lot of noise.
- for Batch Size = 32, the oscillation intensity is smaller but still noticeable
- for Batch Size = 128, the loss curve is smooth.
 
This happens because small batches mean fewer samples per update so the gradient (average) is calculated on a small subset that might not represent the whole dataset well, causing noise (meaning the weights are updated in a direction that does not reflect the true data trend).
 
    
**To clarify**: 

if a subset consists of 8 samples and 2 of them are hard to predict, they will heavily affect the average, meaning the gradient does not represent the actual trend (gradient noise). With a larger subset of 128 samples, those same 2 samples will not affect the overall average (smooth gradient), so the weights are updated in the right direction, making the loss decrease smoothly and represent the true trend of the data.

### Regarding local minima and generalization:

### Why larger batches may converge faster but generalize worse:

When the batches are large, the gradients calculated are stable and they usually settle in a sharp minimum that fits the training data perfectly but is bad for generalization on unseen data.

### When this noise is beneficial (escaping local minima):

The noise introduced by small batches helps the model escape poor local minima. This is considered an advantage if the local minimum was poor so the model is given a chance to search for better minimal this is a disadvantage if the local minimum was actually good, causing instability and preventing the model from properly converging.

## 5. Key Takeaway

The smaller the batch size is, the more it produce noise which can be both an advantage to help escape poor local minima and a disadvantage with good local minima where noise causes instability.