## to improve... for each number take a gaussian standard deviation for the different neurons
## use this gaussian to generate realistic values for the one_hot 
## also train a better NN with more hidden neurons
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import mnist_loader
import training_for_recognition as NN

import random


# --------------------------------------------------
# 1. Inverse network: 10 -> 100 -> 784
# --------------------------------------------------

class InverseNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 500),
            nn.Sigmoid(),
            nn.Linear(500, 784),
            nn.Sigmoid()
        )

    def forward(self, y):
        return self.net(y)


# --------------------------------------------------
# 2. Evaluation
# --------------------------------------------------

def evaluate(model, loader, device):

    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for y, x in loader:
            y, x = y.to(device), x.to(device)

            reconstructed = model(y)
            loss = criterion(reconstructed, x)

            total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


# --------------------------------------------------
# 3. Training
# --------------------------------------------------

def train(model, train_data, val_data,
          epochs=30, batch_size=100, lr=0.001):

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        model.train()

        for y, x in train_loader:

            y, x = y.to(device), x.to(device)

            optimizer.zero_grad()

            reconstructed = model(y)
            loss = criterion(reconstructed, x)

            loss.backward()
            optimizer.step()

        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- Validation MSE: {val_loss:.6f}"
        )

    return model

def one_hot_encode(n):
  if not 0 <= n <= 9:
    raise ValueError("Input must be an integer between 0 and 9.")
  return [random.uniform(0.975, 1) if i == n else random.uniform(0.0, 0.01) for i in range(10)]


# --------------------------------------------------
# 4. Prepare inverse training data
# --------------------------------------------------

training_data, validation_data, test_data = (
    mnist_loader.load_data_wrapper_linear()
)

# Load the already-trained forward classifier
forward_model = NN.model_load(which_model="lin")
forward_model.eval()

with torch.no_grad():

    def make_inverse_dataset(dataset):

        x, _ = dataset.tensors

        # Original classifier output becomes inverse input
        y = forward_model(x)

        # Original image becomes inverse target
        return TensorDataset(y, x)

    inverse_train = make_inverse_dataset(training_data)
    inverse_val = make_inverse_dataset(validation_data)
    inverse_test = make_inverse_dataset(test_data)


# --------------------------------------------------
# 5. Train inverse network
# --------------------------------------------------

inverse_model = InverseNetwork()

inverse_model = train(
    inverse_model,
    inverse_train,
    inverse_val,
    epochs=5,
    batch_size=100,
    lr=0.0005
)


# --------------------------------------------------
# 6. Test
# --------------------------------------------------

test_loader = DataLoader(inverse_test, batch_size=500)

test_mse = evaluate(
    inverse_model,
    test_loader,
    next(inverse_model.parameters()).device
)

print(f"\nFinal Test MSE: {test_mse:.6f}")

device = torch.device("cpu")
inverse_model.to(device)

# --------------------------------------------------
# 7. Generate an image from a test prediction
# --------------------------------------------------

import from_data_to_images

y = torch.tensor(one_hot_encode(7), dtype=torch.float32)

with torch.no_grad():
    x_reconstructed = inverse_model(y.unsqueeze(0)).squeeze(0)

from_data_to_images.figure_from_array(
    x_reconstructed.cpu().numpy()
)

# y, x_original = inverse_test[0]

# print(y.shape)
 

# inverse_model.eval()

# with torch.no_grad():
#     x_reconstructed = inverse_model(y.unsqueeze(0)).squeeze(0)

# print("Reconstruction MSE:",
#       nn.functional.mse_loss(x_reconstructed, x_original).item())

# from_data_to_images.figure_from_array(
#     x_reconstructed.cpu().numpy()
# )

# from_data_to_images.figure_from_array(
#     x_original.cpu().numpy()
# )
