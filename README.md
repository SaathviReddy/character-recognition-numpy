# Character Recognition Neural Network using NumPy

## Project Overview

This project implements a **Character Recognition Neural Network from scratch using NumPy**.

The system recognizes **35 different classes** consisting of:

- 26 uppercase English alphabets: A-Z
- 9 digits: 1-9

The neural network is implemented manually using NumPy without using high-level machine learning frameworks such as TensorFlow, Keras, or PyTorch.

The project includes image preprocessing, neural network training, model evaluation, single-image prediction, confusion matrix analysis, and error analysis.

---

## Features

- Image preprocessing and dataset preparation
- Grayscale image conversion
- Image resizing to 28 × 28 pixels
- Pixel normalization from 0-255 to 0-1
- Flattening images into 784 input features
- 35-class character classification
- Neural network implemented from scratch using NumPy
- ReLU activation function
- Softmax output layer
- Cross-entropy loss
- Backpropagation
- Mini-batch gradient descent
- Training and validation accuracy tracking
- Test dataset evaluation
- Confusion matrix generation
- Incorrect prediction analysis
- Single-image prediction with confidence score
- Trained model saved as `.npz`

---

## Technology Stack

### Programming Language

- Python

### Libraries

- NumPy
- Pillow (PIL)
- Matplotlib

### Development Environment

- Visual Studio Code
- Python Virtual Environment
- Git
- GitHub

---
## Dataset

The dataset contains **3,500 character images**.

There are 35 classes, with 100 images for each class.

### Classes

```text
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
1 2 3 4 5 6 7 8 9
