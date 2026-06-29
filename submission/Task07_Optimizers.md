# Task 07 — Optimizer Comparison Challenge (Selfstudy)

## 1. Objective
Our goal is to compare the convergence speed, and stability of four different optimizers (SGD, SGD with Momentum, Adam, and AdamW) using simple neural network architecture.


## 2. Code Used

```python
# Defining the optimizers with their proper default learning rates
optimizers = {
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
    'SGD+Momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
    'AdamW': tf.keras.optimizers.AdamW(learning_rate=0.001)
}

# Training loop for each optimizer on the exact same 1-hidden-layer model
for name, opt in optimizers.items():
    model = build_model() 
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))
```



## 3. Results

- Adam & AdamW: Fastest convergence, but suffered from overfitting (Val Loss started rising around Epoch 9).  
- SGD: Slowest convergence, no overfitting (Val Loss steadily decreased to 0.1303).  
- SGD+Momentum: Best overall balance (Highest final Val Acc: 97.86%, Best Val Loss: 0.0896) and fast.  

_(Plots showing validation loss and accuracy curves are saved in `results/optimizer_curves/`)_



## 4. Short Analysis

### 1. Convergence Speed and Stability

Metric | Ranking (Highest to Lowest)
---|---
Convergence Speed | Adam ≈ AdamW > SGD+Momentum > SGD
Stability | SGD+Momentum > SGD > AdamW > Adam



### Performance Observations:

Adam & AdamW were the fastest to reach their convergence points in early epochs but both suffered from overfitting since the validation loss started rising around epoch 9 while training accuracy reached 99%.

SGD was the slowest but showed no clear overfitting since the validation loss steadily decreased throughout all 20 epochs.

SGD+Momentum: I suggest that SGD+Momentum is the most balanced optimizer because it combines fast learning with stability of learning. The validation loss decreased quickly at the beginning, then stayed stable around a low value without increasing, and it achieved a high final validation accuracy (98.44%).



### 2. How Each Optimizer Navigates the Loss Landscape

**SGD**: Takes consistent, equal, stable steps for all weights and it avoids overfitting but is slow to converge because it specifies a common learning rate for all weights, unable to determine which parameters need larger updates. _(Slow + Good Generalization)_


**SGD+Momentum:** It keeps the fixed learning rate from SGD so it avoids the overfitting seen in Adam/AdamW, but adds momentum to improve speed where the momentum accumulates gradients in consistent directions, leading to larger steps and a more direct path to the minimum and for generalization It is excellent since the validation loss dropped rapidly at first, then stabilized smoothly around its lowest value **with only small going up and down** so it is the best balance in our experiment _(Fast + Good Generalization)_.

**Adam:** Gives each weight its own learning rat making it very fast but in simple models this becomes a disadvantage since unlike SGD which naturally slows down when it reaches a good minimum because gradients become small, Adam prevents this slowdown and makes small gradients large to keep pushing the training loss down, meaning it keeps updating the weights aggressively even when the model has already learned enough, leading to overfitting _(Fast + Overfitting)_.

**AdamW:** It is exactly like Adam but applies weight decay to reduce overfitting but in our simple model, it behaved almost like Adam because the model is too simple with few parameters for AdamW to make a noticeable difference, so it still suffered from the same overfitting _(Fast + Overfitting)_.


### 3. Why Adam Often Outperforms Classical Optimizers

From the results we might think that SGD+Momentum is always the best, but this is not the case since Adam Often outperforms classical optimizers like SGD in more complex tasks and here are the reasons:

- **Adaptive Learning Rates:** In neural networks, different weights need different adjustments, SGD uses the same learning rate for all weights so it either converges slowly when the rate is too small or oscillates when it is too large but Adam solves this by giving each weight its own step size where weights with large gradients get smaller steps to avoid overshooting and weights with small gradients get larger steps to avoid getting stuck in flat regions (small gradients often indicate flat areas where the model needs bigger steps to make progress) so all weights are updated at their suitable speeds making the model faster and and more efficient than SGD, especially in deep networks.

- **Built-in Momentum:** Adam remembers past gradients and uses them to keep moving in a smooth, general direction, combining speed with direction.

So to describe Adam in complex tasks, it is fast, adaptive and designed to be stable, which is why it is the default choice for most neural network applications.


This might seem like a contradiction to our experiment! In our simple 1-hidden-layer model, Adam's ability to keep pushing the training loss became a disadvantage (causing overfitting), but in complex tasks where the model actually needs to learn millions of difficult patterns, this same ability makes it the the best.

Because our network was simple, Adams  adjustments foe every single wiegt werent needed and only caused overfitting, so its biggest strength is considered a weakness in such cases.

From this, we can conclude that optimizer selection depends entirely on the task and architecture, Adam is the best choice for complex networks while simpler optimizers like SGD+Momentum are better for simple tasks.


## 5. Key Takeaway

Optimizer selection depends on model complexity whwre adaptive optimizers like Adam are the best in deep networks but can cause overfitting in simple models, where SGD+Momentum provides the best balance of speed and generalization.
