# Task 02 — Custom Image Generalization Test

## 1. Objective

Our goal is to test the model on a handwritten digit that we created ourselves (not from the MNIST dataset) and see whether the model can correctly classify it or not. I draw the digit shifted to the right with very heavy strokes and added some random dots added as noise to make the prediction more challenging, I then created a second digit that fits the models training data (centered, clean strokes, no noise) and tested it to confirm that the model works correctly when the input matches the training distribution.

## 2. Code Used

```python
from PIL import Image
img = Image.open("results/predictions/Task02/handwritten_5_original.png")

img = img.convert('L')
img = img.resize((28, 28))

img_array = np.array(img)

img_array = 255 - img_array
img_array = img_array / 255.0

img_reshaped = img_array.reshape(1, 28, 28)

prediction = model.predict(img_reshaped)
print("Predicted Label:", np.argmax(prediction))
print("Confidence:", np.max(prediction)*100)
print("True Label: 5")

plt.imshow(img_array, cmap="gray")
plt.show()
```

## 3. Results

- **First Input Digit:** 5
- **Predicted Digit:** 2
- **Confidence Rate:** 98.76

- **Second Input Digit**: 5 
- **Predicted Digit**: 5 
- **Confidence Rate**: 85.48

Note that for the true label 5, when the model predicted it false , its Confidence rate was high (98.76) while when it predicted it correctly, its Confidence rate was lower (85.48)

_(Images showing the handwritten digit — the original and the processed version — are saved in `results/predictions/`)_


## 4. Short Analysis

The model failed in classifying the digit since it predicted 2 instead of 5 and here are the potential reasons:

1. **Distribution shift**: Our digit is shifted to the right while MNIST trained on centered digits, so a shifted digit strange to the model so it fails in classifying it.

2. **Noise**: Our digit contains random dots around it which confused the model while MNIST trained on clean images with no noise, so a  digit with noise is slso strange to the model and the model did not learn how to handle it so it fails in classifying it also.

3. **Lack of augmentation**: The test image we created looks  different from what the model saw during training so if the model trained on shifted digits with noise, it would give correct results.


### How Data Augmentation Could Fix This?

Data augmentation (Shift, Rotate, Noise) is an effective technique to solve this problem raised, by training the model on versions of the training data which helps it learn the general patterns of each digit rather focusing on details.

- Shift (width_shift / height_shift): will teach the model that digits don't have to be centered, they could be shifted to the right or left, so the shift problem will be solved.
- Rotation : will teach the model how to handle slight rotations in handwriting style, so it is more realistic and close to handwritten digits.
- Noise: will add small random noise during training to teach the model to ignore dots around the digits like the ones we added, so it is no more confused with the dots.

so training on augmented data teachs the model learns that the general shape of the digit is what matters and teaches the model to focus on it, not on the position, rotation, or noise which made the model fails in prediction


To confirm that the model works correctly, we tested a second handwritten digit that fits the MNIST distribution (centered, clean strokes, no noise). The model classified it correctly, which proves that the failure in the first test was due to distribution shift and noise not because the model is bad.


## 5. Key Takeaway

Neural networks work well when test data is similar to the data they trained on. If we fed the network with inputs that have different features like shift, noise and different stroke then the model will  fail, this shows us how data augmentation is important.
