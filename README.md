# Foundations of Deep Learning: Neural Networks, Optimization, and Regularization — MNIST Classification

## Project Overview

This project is a comprehensive learning module that explores neural networks through hands-on experiments with the MNIST dataset (a collection of labeled images of handwritten digits (0–9) used for training and evaluating models.). It covers the full workflow from data loading and model building to optimization, regularization, and training dynamics analysis.

## Notebook Sections

### Section 1 — Building & Training a Neural Network
A simple neural network is built using Keras: Flatten input (28×28) → Dense (128, ReLU) → Dense (10, Softmax). The model is compiled with Adam optimizer and trained on MNIST to create a baseline understanding of forward/backward passes.

### Section 2 — Data Splitting, Regularization & Early Stopping
The dataset is split into train (55,000) / validation (5,000) / test (10,000), we introduces Dropout and EarlyStopping to improve generalization and prevent overfitting.

### Section 3 — Student Tasks
Experiments that shows different aspects of training:
- **Task 01** — Deep Prediction Analysis on test samples
- **Task 02** — Custom Handwritten Digit Generalization Test
- **Task 03** — Epoch-Based Learning Curve Exploration
- **Task 04** — EarlyStopping Behavior Analysis
- **Task 05** — Dropout Ablation Study
- **Task 06** — L2 Regularization Experiment
- **Task 07** — Optimizer Comparison Challenge
- **Task 08** — Batch Size & Gradient Noise Experiment
- **Task 09** — Activation Function Swap

## How to Run

1. Make sure you have Python 3.7+ and the following packages installed:
   ```bash
   pip install tensorflow numpy matplotlib pillow
   ```

2. Open `notebook.ipynb` in Jupyter Notebook or Google Colab.

3. Run all cells from top to bottom — the notebook is designed to run sequentially without errors.

## Note on Reproducibility

To ensure fair comparison, random seeds are set before training each optimizer (`np.random.seed(42)` and `tf.random.set_seed(42)`). This guarantees that all start from the exact same initial weights, so any difference in results is due to the optimizer itself not random initialization.


## Repository Structure

```
project/
├── notebook.ipynb
├── README.md
├── results/
│   ├── predictions/
│   ├── loss_curves/
│   └── optimizer_tests/
└── submission/
    ├── Task01_PredictionAnalysis.md
    ├── Task02_CustomDigit.md
    ├── Task03_Epochs.md
    ├── Task04_EarlyStopping.md
    ├── Task05_Dropout.md
    ├── Task07_Optimizers.md
    ├── Task08_BatchSize.md
    └── Task09_Activations.md
```

## Sample Results

### Task 01 — Predictions

All three test samples (digits 0, 1, 5) were correctly classified by the model.

### Task 02 — Predictions

The handwritten sample(5) was misclassified as 1 by the model .

### Task 03 — Overfitting with More Epochs
| Epochs | Train Loss | Val Loss | Status |
|--------|-----------|----------|--------|
| 5 | 0.0638 | 0.0685 | No overfitting |
| 10 | 0.0314 | 0.0827 | Overfitting started |
| 20 | 0.0159 | 0.0840 | Severe overfitting |

### Task 04 — EarlyStopping
| Config | Stopped at | Best Val Loss | Best at Epoch |
|--------|-----------|--------------|---------------|
| patience=3 (Adam) | Epoch 8 | 0.0676 | 5 |
| patience=5 (Adam) | Epoch 11 | 0.0637 | 6 |
| SGD (no early stop) | Completed 20 | 0.0991 | — |

### Task 05 — Dropout Impact
| Dropout | Train Loss | Val Loss | Gap |
|---------|-----------|----------|-----|
| 0.0 | 0.0041 | 0.1045 | 0.1004 (Severe) |
| 0.1 | 0.0164 | 0.0833 | 0.0669 (Moderate) |
| 0.3 | 0.0435 | 0.0850 | 0.0415 (Generalizing) |

### Task 07 — Optimizer Comparison
| Optimizer | Best Val Loss | Val Accuracy | Note |
|-----------|--------------|-------------|------|
| SGD | 0.0997 | 97.40% | Slowest, no overfitting |
| SGD+Momentum | 0.0577 | 98.52% | Best balance |
| Adam | 0.0718 | 98.06% | Fast but overfitting |
| AdamW | 0.0713 | 97.92% | Fast but overfitting |


### Task 08 — Batch Size Effect
| Batch Size | Val Loss | Generalization |
|-----------|----------|----------------|
| 8 | 0.1268 | Worst (high noise) |
| 32 | 0.0916 | Moderate |
| 128 | 0.0783 | Best (smooth gradients) |

### Task 09 — Activation Functions
| Activation | Train Loss | Val Loss | Note |
|-----------|-----------|----------|------|
| ReLU | 0.0141 | 0.0898 | Clear overfitting |
| Tanh | 0.0181 | 0.0772 | Smallest gap |
| GELU | 0.0128 | 0.0926 | Fastest train, highest val loss |

For full plots and detailed analysis, refer to the `results/` folder and `submission/` folder.




