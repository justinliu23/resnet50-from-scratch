# residual-network-from-scratch

## Table of Contents
1. [Introduction](#introduction)
2. [Installation Instructions](#installation-instructions)
    - [Prerequisites](#prerequisites)
    - [Dependencies](#dependencies)
3. [Usage](#usage)
4. [Features](#features)
5. [Configuration](#configuration)

<a name="introduction"></a>
## 1. Introduction

This project demonstrates the implementation of a Residual Network (ResNet), a type of deep neural network that utilizes skip connections to address the problem of vanishing gradients in very deep networks. The project walks through building the key components of ResNet, including the identity block and the convolutional block, and culminates in the construction of a ResNet50 model, a 50-layer deep network.

The ResNet architecture is notable for its ability to enable the training of very deep networks by allowing gradients to flow directly through the network by skipping layers. This project showcases the step-by-step process of implementing these blocks, constructing the full ResNet50 model, and training it on a dataset.

<a name="installation-instructions"></a>
## 2. Installation Instructions

### Prerequisites

Before running the project, ensure you have the following software installed:
- Python 3.7+
- Jupyter Notebook
- TensorFlow 2.x

### Dependencies

The project relies on several Python libraries. You can install the required packages using pip:

- TensorFlow: `pip install tensorflow`
- NumPy: `pip install numpy`
- SciPy: `pip install scipy`
- Matplotlib: `pip install matplotlib`

To install all dependencies at once, you can use the following command:

```bash
pip install tensorflow numpy scipy matplotlib
```

<a name="usage"></a>
## 3. Usage

The usage section provides instructions on how to run the Jupyter notebooks included in this repository. These notebooks guide you through the process of building and training a Residual Network using TensorFlow and Keras.

### Step-by-Step Workflow:

1. **Import Required Libraries:**
   - Import necessary libraries including TensorFlow, Keras, NumPy, SciPy, and Matplotlib.

2. **Load and Preprocess Data:**
   - Load the dataset and preprocess it by normalizing the image vectors and converting labels to one-hot matrices.

3. **Define Identity and Convolutional Blocks:**
   - Implement the identity block and convolutional block, the building blocks of ResNet. These blocks include convolutional layers, batch normalization, and activation functions.

4. **Build the ResNet50 Model:**
   - Construct the full ResNet50 architecture using the blocks defined earlier. The model is composed of multiple stages, each containing convolutional and identity blocks.

5. **Compile the Model:**
   - Compile the model with an optimizer (Adam), a loss function (categorical crossentropy), and metrics (accuracy).

6. **Train the Model:**
   - Train the model on the training dataset using a specified number of epochs and batch size.

7. **Evaluate the Model:**
   - Evaluate the model on the test dataset to determine its loss and accuracy.

8. **Load Pre-trained Model (Optional):**
   - Load a pre-trained ResNet50 model and evaluate its performance on the test dataset.

<a name="features"></a>
## 4. Features

- **Residual Connections:** Implement skip connections to prevent vanishing gradients, allowing the training of very deep networks.
- **Custom Blocks:** Build identity and convolutional blocks that form the foundation of the ResNet architecture.
- **ResNet50 Architecture:** Construct a 50-layer deep residual network.
- **Data Preprocessing:** Normalize image data and convert labels to one-hot encoded vectors for training.
- **Model Evaluation:** Evaluate the trained model's performance on a test dataset and compare it with a pre-trained model.

<a name="configuration"></a>
## 5. Configuration

The configuration section provides details on how to customize the model and training process according to specific requirements.

### Modifiable Parameters:

1. **Input Shape:**
   - The input shape of the images can be adjusted based on the dataset being used. The default is `(64, 64, 3)`.

2. **Number of Classes:**
   - The number of output classes can be modified to match the number of categories in your dataset. The default is `6`.

3. **Learning Rate:**
   - Adjust the learning rate in the Adam optimizer to control the speed of model training. The default is `0.00015`.

4. **Batch Size and Epochs:**
   - Modify the batch size and number of epochs during model training to fit the computational resources and dataset size.

5. **Pre-trained Model:**
   - Optionally, load and evaluate a pre-trained ResNet50 model to compare its performance with the custom-built model.
