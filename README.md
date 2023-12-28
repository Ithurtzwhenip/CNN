# Convolutional Neural Network (CNN) for Cat vs Dog Classification

This repository contains the code for a Convolutional Neural Network (CNN) designed to classify images as either cats or dogs. The model is implemented using the Keras library with TensorFlow backend.

## Model Architecture

The CNN architecture consists of the following layers:

1. **Convolutional Layers:**
   - Two sets of convolutional layers with ReLU activation functions.
   - The first layer processes input images with a shape of (64, 64, 3).
   - Each convolutional layer is followed by max-pooling layers to reduce spatial dimensions.

2. **Flattening Layer:**
   - Flattens the output from the convolutional layers into a one-dimensional vector.

3. **Fully Connected Layers:**
   - Two dense (fully connected) layers with ReLU activation.
   - The final output layer uses a sigmoid activation function for binary classification.

4. **Compiling the Model:**
   - The model is compiled using the Adam optimizer, binary cross-entropy loss function, and accuracy as the evaluation metric.

## Image Augmentation and Preprocessing

The training dataset is augmented using the ImageDataGenerator from Keras. Augmentation techniques include rescaling, shearing, zooming, and horizontal flipping. The test set is rescaled for consistency.

## Dataset

The model is trained and evaluated on a dataset of cat and dog images. The dataset is divided into training and test sets.

## Training the Model

The CNN is trained using the `fit_generator` function for 25 epochs. Training and validation steps are determined based on the number of batches.

## Making Predictions

After training, the model can make predictions on new images. The provided script loads an image, preprocesses it, and predicts whether it's a cat or a dog.

## Dependencies

- Keras
- TensorFlow
- NumPy

## Usage

Clone the repository and run the provided Python script. Ensure you have the required dependencies installed.

```bash
python cnn.py
```

---
