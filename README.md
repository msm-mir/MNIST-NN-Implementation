# MNIST NN Implementation

This project contains an end-to-end, object-oriented implementation of a fully connected Artificial Neural Network (ANN) designed to classify handwritten digits from the **MNIST dataset**. It is built entirely from scratch using fundamental mathematical equations.

---

## Key Features

- **Pure NumPy Implementation:** All core components, including matrix operations, weight initialization, forward propagation, backpropagation, and weight updates, are coded manually.
- **Activation Functions:** Hand-calculated implementations of **Sigmoid** and **ReLU** alongside their exact mathematical derivatives.
- **Loss Function:** Supports **Cross-Entropy Loss** for multi-class optimization.
- **Modular Architecture:** Supports customizable layer sizes, numbers of hidden layers, and configurable hyperparameters.
- **Comprehensive Hyperparameter Analysis:** Built-in experimentation infrastructure to test and visualize the effects of Learning Rates, Epochs, and Batch Sizes.
- **Advanced Optimizations Implemented:** Includes Mini-Batch Gradient Descent, Dropout regularization, Early Stopping, an automated Save/Load state system, and the advanced Adam Optimizer.

---

## Dataset Information

The model is trained and evaluated on the classic **MNIST dataset**:
- **Training Set:** 60,000 images
- **Testing Set:** 10,000 images
- **Image Dimensions:** $28 \times 28$ pixels
- **Format:** Grayscale intensities ranging from 0 to 255
- **Target Classes:** 10 classes corresponding to digits 0-9.

---

## Project Structure

```
├── src/
│   └── src.py
├── .gitignore
└── README.md
```
