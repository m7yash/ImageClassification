# ImageClassification

## Overview

This project contains code for training and evaluating neural networks on the CIFAR-10 dataset for image classification. The project demonstrates both a linear model and more complex architectures using ReLU and Conv2D layers.

## Features

- Dataset: CIFAR-10
- Models: Logistic Regression, Fully Connected NN, CNN
- Tools: PyTorch, Matplotlib
- Training and validation loops
- Hyperparameter tuning

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- tqdm

## Setup

1. Clone this repository
2. Install the required packages
3. Run the script

## Usage

The main functionalities of this script include:

- Loading the CIFAR-10 dataset
- Data preprocessing and splitting into train, validation, and test sets
- Defining various neural network architectures
- Training the models
- Hyperparameter tuning
- Evaluation on test set
- Visualization of accuracies

The script contains helper functions to perform each of these steps. For instance, to train a linear model:

```python
best_lr = parameter_search(train_loader, val_loader, linear_model)
model = linear_model()
optimizer = SGD(model.parameters(), best_lr)
train(model, optimizer, train_loader, val_loader, 20)
```

For evaluation:

```python
test_loss, test_acc = evaluate(model, test_loader)
```

## Output

Training and validation accuracies are plotted using matplotlib. The best hyperparameters are printed, and the test accuracies for the best models are displayed.
