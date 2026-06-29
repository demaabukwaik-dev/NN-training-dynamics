# Task 09 — Activation Function Swap (ReLU vs Tanh vs GELU)

## 1. Objective

Our goal is to analyze how replacing the activation function affects the model's training amd focusing on how each function handles gradient flow, which ones risk vanishing gradients, and why GELU is preferred in Transformers while ReLU is preferred in simple MLPs.

## 2. Code Used

```python
# Train the model with different activation functions

histories = [] # sroring the histories of the models with different activation functions in a list to plot later
activation_functions_list = ['relu','tanh', 'GELU'] # the activation functions we want to train the model with

# Train the model with different activation functions
for a_func in activation_functions_list:
    print(f"\n--- Training with  {a_func} activation ---")

    # build model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=a_func),
        keras.layers.Dropout(0.1),
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
- ReLU: Train Loss: 0.0273 | Val Loss: 0.0829 (Clear Overfitting)
- Tanh: Train Loss: 0.0189 | Val Loss: 0.0848 (Smallest Generalization Gap)
- GELU: Train Loss: 0.0134 | Val Loss: 0.0936 (Clear Overfitting)


_(Plots showing the loss curves for all runs are saved in `results/loss_curves/`)_



## 4. Short Analysis

### ReLU & GeLU

- ReLU
passes positive gradients so ( no vanishing gradients when going back through the layers), and it neglects negative gradients (outputs them 0), this is good because it makes the model focuses on features that really affect the prediction, if a neuron receives too many negative inputs, this may cause “dying ReLU” where some neurons stop learning because of this efficiency and simplicity. ReLU remains the preferred choice for standard MLP and CNN models where deep layers don't need GELU's complex math. 


- **GeLU:**
works excactly like ReLU for positive gradients but for negative values it allows for slightly small gradients instesd of 0, this solves the problem raised in relu (dying relu) in deep networks, so it is the best choice for the deep layers and attention mechanisms used in Transformers since it helps them learn better in deep layers and keeps small important information that helps understand context in sentences, for example a word 'not' is very small but can completely change the meaning so if a model ignored it, the whole context will be destroyed.



### Tanh & Softsign:

They are similar as both work by compressing input numbers into fixed range which is small (-1,1) causing the vanishing gradient problem in deep networks, but they do it at different speeds.

During backpropagation, each layer receives a gradient from the layer above it.

This incoming gradient is multiplied by the derivative of the activation function (Tanh)
computed using the layer's own Z value (Z = W·X + b).

The derivative of tanh behaves like this:
At Z = 0 → tanh'(Z) = 1 (maximum gradient flow)
As Z moves far from 0 (positive or negative) → tanh'(Z) → 0

In early layers (closer to output), Z values are often closer to 0,
so tanh'(Z) is relatively large → gradients are preserved → good learning.

In deeper layers (closer to input), Z values often fall into saturation regions,
so tanh'(Z) becomes very small.

This causes the gradient to shrink layer after layer:
gradient → small × small × small → approaches 0

This is what we call "Vanishing Gradient",
where early layers receive almost no learning signal and stop updating properly.


- **Tanh (Fast Vanishing):**
Tanh compresses numbers in very aggresive way, its maximum gradient is 1 (at zero), but it drops to near zero very quickly as numbers move away from zero. In deep networks, backpropagation multiplies these small numbers together repeatedly. Because the gradient hits near-zero so fast, the early layers receive almost no error signal and stop learning, while the output layers still learn fine.

- **Softsign (Slow Vanishing):**
Softsign is a gentler version of Tanh as the numbers going far from zero , Softsign does not maps them directly to 1 and -1 like Tanh does, so their derivatives are quickly 0 and only few layers back are learned good. Instead it maps the numbers into 1 and -1 gradually so their derivative at the end reach 0 but not directly, this enables more earlier layers to learn well.

### Why Tanh Had the Smallest Generalization Gap
Among the three tested activations, Tanh achieved the smallest gap between train and validation loss this is because Tanh compresses values between (-1, 1), which naturally prevents the network from becoming overconfident during training and reduces overfitting.

## 5. Key Takeaway

While GELU provides the solution for dying relu problem in deep networks like Transformers, ReLU remains the most efficient and preferred choice for standard MLPs and CNNs because simple networks do not suffer from the gradient vanishing issues that require GELU's smooth complexity. Tanh, despite its vanishing gradient risk in very deep networks but here in our model it showed the best generalization (lowest gap) because our network is not deep (1 layer) and due to its compression.