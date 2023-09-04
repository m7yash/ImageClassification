"""Original file is located at
    https://colab.research.google.com/drive/1Ae-SQZA4o2CM9OJNyHbThJNo4wNKHagP
"""

import torch
from torch import nn

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# assert torch.cuda.is_available(), "GPU is not available"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

batch_size = 128

train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])

# Create separate dataloaders for the train, test, and validation set
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)

imgs, labels = next(iter(train_loader))
print(f"A single batch of images has shape: {imgs.size()}")
example_image, example_label = imgs[0], labels[0]
c, w, h = example_image.size()
print(f"A single RGB image has {c} channels, width {w}, and height {h}.")

# This is one way to flatten our images
batch_flat_view = imgs.view(-1, c * w * h)
print(f"Size of a batch of images flattened with view: {batch_flat_view.size()}")

# This is another equivalent way
batch_flat_flatten = imgs.flatten(1)
print(f"Size of a batch of images flattened with flatten: {batch_flat_flatten.size()}")

# The new dimension is just the product of the ones we flattened
d = example_image.flatten().size()[0]
print(c * w * h == d)

# View the image
t = torchvision.transforms.ToPILImage()
plt.imshow(t(example_image))

# These are what the class labels in CIFAR-10 represent. For more information,
# visit https://www.cs.toronto.edu/~kriz/cifar.html
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
           "horse", "ship", "truck"]
print(f"This image is labeled as class {classes[example_label]}")


def linear_model() -> nn.Module:
    """Instantiate a linear model and send it to device."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(d, 10)
    )
    return model.to(DEVICE)


def train(
        model: nn.Module, optimizer: SGD,
        train_loader: DataLoader, val_loader: DataLoader,
        epochs: int = 20
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains a model for the specified number of epochs using the loaders.
  
    Returns: 
      Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
    """

    loss = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for e in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # Main training loop; iterate over train_loader. The loop
        # terminates when the train loader finishes iterating, which is one epoch.
        for (x_batch, labels) in train_loader:
            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            labels_pred = model(x_batch)
            batch_loss = loss(labels_pred, labels)
            train_loss = train_loss + batch_loss.item()

            labels_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(labels_pred_max == labels)
            train_acc = train_acc + batch_acc.item()

            batch_loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (batch_size * len(train_loader)))

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for (v_batch, labels) in val_loader:
                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
                labels_pred = model(v_batch)
                v_batch_loss = loss(labels_pred, labels)
                val_loss = val_loss + v_batch_loss.item()

                v_pred_max = torch.argmax(labels_pred, 1)
                batch_acc = torch.sum(v_pred_max == labels)
                val_acc = val_acc + batch_acc.item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc / (batch_size * len(val_loader)))

    return train_losses, train_accuracies, val_losses, val_accuracies

def parameter_search(train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_fn: Callable[[], nn.Module]) -> float:
    """
    Parameter search for our linear model using SGD.
  
    Args:
      train_loader: the train dataloader.
      val_loader: the validation dataloader.
      model_fn: a function that, when called, returns a torch.nn.Module.
  
    Returns:
      The learning rate with the least validation loss.
    """
    num_iter = 10  # This will likely not be enough for the rest of the problem.
    best_loss = torch.inf
    best_lr = 0.0

    lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_iter)

    for lr in lrs:
        print(f"trying learning rate {lr}")
        model = model_fn()
        optim = SGD(model.parameters(), lr)

        train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=20
        )

        if min(val_loss) < best_loss:
            best_loss = min(val_loss)
            best_lr = lr

    return best_lr

# best_lr = parameter_search(train_loader, val_loader, linear_model)

# model = linear_model()
# optimizer = SGD(model.parameters(), best_lr)

# # We are only using 20 epochs for this example. You may have to use more.
# train_loss, train_accuracy, val_loss, val_accuracy = train(
#     model, optimizer, train_loader, val_loader, 20)


# epochs = range(1, 21)
# plt.plot(epochs, train_accuracy, label="Train Accuracy")
# plt.plot(epochs, val_accuracy, label="Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.title("Logistic Regression Accuracy for CIFAR-10 vs Epoch")
# plt.show()

def evaluate(
        model: nn.Module, loader: DataLoader
) -> Tuple[float, float]:
    """Computes test loss and accuracy of model on loader."""
    loss = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (batch, labels) in loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            y_batch_pred = model(batch)
            batch_loss = loss(y_batch_pred, labels)
            test_loss = test_loss + batch_loss.item()

            pred_max = torch.argmax(y_batch_pred, 1)
            batch_acc = torch.sum(pred_max == labels)
            test_acc = test_acc + batch_acc.item()
        test_loss = test_loss / len(loader)
        test_acc = test_acc / (batch_size * len(loader))
        return test_loss, test_acc


# test_loss, test_acc = evaluate(model, test_loader)
# print(f"Test Accuracy: {test_acc}")

def nn_a_model(M) -> nn.Module:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, M),
        nn.ReLU(),
        nn.Linear(M, 10)
    )
    return model.to(DEVICE)


def nn_b_model(M, N, k) -> nn.Module:
    dim = (33 - k) // N
    model = nn.Sequential(
        nn.Conv2d(3, M, k),
        nn.ReLU(),
        nn.MaxPool2d(N, N),
        nn.Flatten(),
        nn.Linear(M * (dim ** 2), 10)
    )
    return model.to(DEVICE)


def nn_parameter_search_a(train_loader: DataLoader,
                          val_loader: DataLoader,
                          model_fn: Callable[[], nn.Module]) -> Tuple:
    m_range = [100, 200, 300, 400, 700, 1000]
    lr_range = [10 ** (-4), 10 ** (-3), 10 ** (-2)]
    momentum = 0.9

    lrs = []
    ms = []
    accuracies = []

    for lr in lr_range:
        for m in m_range:
            print(f"lr: {lr}, M: {m}")
            model = model_fn(m)
            optimizer = SGD(model.parameters(), lr, momentum)
            train_loss, train_accuracy, val_loss, val_accuracy = train(
                model, optimizer, train_loader, val_loader, 30)
            lrs.append(lr)
            ms.append(m)
            accuracies.append(val_accuracy[-1])
            print("\t accuracy: ", accuracies[-1])
    return lrs, ms, accuracies


a_lrs, a_ms, a_accuracies = nn_parameter_search_a(train_loader, val_loader, nn_a_model)

print("a_lrs: ", a_lrs)
print("a_ms: ", a_ms)
print("a_accuracies: ", a_accuracies)

a_best_indices = sorted(range(len(a_accuracies)), key=lambda i: a_accuracies[i])[-3:]
a_best_lrs = [a_lrs[i] for i in a_best_indices]
a_best_ms = [a_ms[i] for i in a_best_indices]
a_best_accuracies = [a_accuracies[i] for i in a_best_indices]

print("best lrs: ", a_best_lrs)
print("best ms: ", a_best_ms)
print("best accuracies: ", a_best_accuracies)

a_best_models = []
epochs = 30
epoch_runs = range(1, epochs + 1)

for i in range(len(a_best_ms)):
    model = nn_a_model(a_best_ms[i])
    optimizer = SGD(model.parameters(), a_best_lrs[i], 0.9)
    train_loss, train_accuracy, val_loss, val_accuracy = train(
        model, optimizer, train_loader, val_loader, epochs)
    a_best_models.append(model)
    plt.plot(epoch_runs, val_accuracy, '--',
             label="Validation for M being " + str(a_best_ms[i]) + " and learning rate being " + str(a_best_lrs[i]))
    plt.plot(epoch_runs, train_accuracy,
             label="Training for M being " + str(a_best_ms[i]) + " and learning rate being " + str(a_best_lrs[i]))
plt.axhline(y=0.5, color='red', linestyle='-')
plt.title("Accuracies Image Classification on CIFAR-10 with Part A Network Architecture")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()

for i in range(len(a_best_models)):
    test_loss, test_acc = evaluate(a_best_models[i], test_loader)
    print("Test Accuracy for Part A Network Architecture (M = " + str(a_best_ms[i]) + ", lr = " + str(
        a_best_lrs[i]) + "): " + str(test_acc))


def nn_parameter_search_b(train_loader: DataLoader,
                          val_loader: DataLoader,
                          model_fn: Callable[[], nn.Module]) -> Tuple:
    m_range = [100, 200, 300, 400, 700, 1000]
    lr_range = [10 ** (-4), 10 ** (-3), 10 ** (-2)]
    momentum = 0.9

    lrs = []
    ms = []
    accuracies = []

    n = 14
    k = 5

    for lr in lr_range:
        for m in m_range:
            print(f"lr: {lr}, M: {m}")
            model = model_fn(m, n, k)
            optimizer = SGD(model.parameters(), lr, momentum)
            train_loss, train_accuracy, val_loss, val_accuracy = train(
                model, optimizer, train_loader, val_loader, 5)
            lrs.append(lr)
            ms.append(m)
            accuracies.append(val_accuracy[-1])
            print("\t accuracy: ", accuracies[-1])
    return lrs, ms, accuracies


b_lrs, b_ms, b_accuracies = nn_parameter_search_b(train_loader, val_loader, nn_b_model)

print("b_lrs: ", b_lrs)
print("b_ms: ", b_ms)
print("b_accuracies: ", b_accuracies)

b_best_indices = sorted(range(len(b_accuracies)), key=lambda i: b_accuracies[i])[-3:]
b_best_lrs = [b_lrs[i] for i in b_best_indices]
b_best_ms = [b_ms[i] for i in b_best_indices]
b_best_accuracies = [b_accuracies[i] for i in b_best_indices]

print("best lrs: ", b_best_lrs)
print("best ms: ", b_best_ms)
print("best accuracies: ", b_best_accuracies)

n = 14
k = 5

b_best_models = []
epochs = 30
epoch_runs = range(1, epochs + 1)

for i in range(len(b_best_ms)):
    model = nn_b_model(b_best_ms[i], n, k)
    optimizer = SGD(model.parameters(), b_best_lrs[i], 0.9)
    train_loss, train_accuracy, val_loss, val_accuracy = train(
        model, optimizer, train_loader, val_loader, epochs)
    b_best_models.append(model)
    plt.plot(epoch_runs, val_accuracy, '--',
             label="Validation for M being " + str(b_best_ms[i]) + " and learning rate being " + str(b_best_lrs[i]))
    plt.plot(epoch_runs, train_accuracy,
             label="Training for M being " + str(b_best_ms[i]) + " and learning rate being " + str(b_best_lrs[i]))
plt.axhline(y=0.65, color='red', linestyle='-')
plt.title("Accuracies Image Classification on CIFAR-10 with Part B Network Architecture")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()

for i in range(len(b_best_models)):
    test_loss, test_acc = evaluate(b_best_models[i], test_loader)
    print("Test Accuracy for Part B Network Architecture (M = " + str(b_best_ms[i]) + ", lr = " + str(
        b_best_lrs[i]) + "): " + str(test_acc))