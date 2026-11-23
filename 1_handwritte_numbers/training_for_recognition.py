import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import tempfile
from pathlib import Path

# library defined by Michael Nielsen and adapted by me
import mnist_loader

#### create the highly sucessful Nielsen's network from https://neuralnetworksanddeeplearning.com/chap6.html 
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Conv (1 -> 20 channels, 5x5 kernel) + MaxPool (2x2)
        # Input: (batch, 1, 28, 28) -> Conv: (batch, 20, 24, 24) -> Pool: (batch, 20, 12, 12)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 2: Conv (20 -> 40 channels, 5x5 kernel) + MaxPool (2x2)
        # Input: (batch, 20, 12, 12) -> Conv: (batch, 40, 8, 8) -> Pool: (batch, 40, 4, 4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 3: Fully Connected (40 * 4 * 4 -> 100)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(40 * 4 * 4, 100),
            nn.ReLU()
        )
        
        # Layer 4: Output Layer (100 -> 10)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

#### create the simple one hidden layer sucessful Nielsen's network from https://neuralnetworksanddeeplearning.com/chap3.html
class NetworkSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 100),  # Expects (batch, 784) directly
            nn.Sigmoid(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.net(x)

#### defining functions to check the model's accuracy and train it
def evaluate(model, loader, device):
    model.eval()  # Set model to evaluation mode
    correct, total = 0, 0
    with torch.no_grad():  # Disable gradient calculation for speed
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
    return 100.0 * correct / total  

def train_and_evaluate(model, train_data, val_data, test_data, epochs=60, mini_batch_size=10, lr=0.1, w_decay=1e-5):
    # Use GPU if available via defining the device dynamically
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model.to(device)
    print("Model is in ", device)

    # Create DataLoaders for mini-batching
    train_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=mini_batch_size)
    test_loader = DataLoader(test_data, batch_size=mini_batch_size)

    # Loss and Optimizer (CrossEntropyLoss includes Softmax)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=w_decay)

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()          # Clear previous gradients
            outputs = model(x_batch)       # Forward pass
            loss = criterion(outputs, y_batch) # Compute loss
            loss.backward()                # Backpropagation
            optimizer.step()               # Update weights

        # Validation check at the end of each epoch
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {val_acc:.2f}%")

    # Final Test evaluation
    test_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

#### a function to save the model safely  
def save_weights_model_cpu_safely(model, fil_name="conv"):
    # Construct file name
    if fil_name == "conv":
        prefix = "conv"
    else:
        prefix = "lin"
    suffix = "_model.pt"
    filename = f"{prefix}{suffix}"
    # Get the directory where this Python script is located
    script_dir = Path(__file__).resolve().parent
    # Join it with your file path
    file_path = script_dir / "Saved_models" / filename 

    # 1. Record original device so we can restore it if training continues
    original_device = next(model.parameters()).device
    
    # 2. Ensure the target directory exists
    target_dir = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(target_dir, exist_ok=True)
    
    tmp_path = None
    try:
        # 3. Move model to CPU before saving
        model.to("cpu")
        
        # 4. Create a temporary file in the SAME directory as the final file
        # (Required so os.replace() performs an instant, atomic rename)
        with tempfile.NamedTemporaryFile(dir=target_dir, delete=False, suffix=".tmp") as tmp_file:
            tmp_path = tmp_file.name
            torch.save(model.state_dict(), tmp_file)
            
            # Flush Python and OS buffers to ensure data is physically written to disk
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
            
        # 5. Atomically rename temp file to final destination (overwrites safely)
        os.replace(tmp_path, file_path)
        print("File save at ", file_path)
        
    except Exception as e:
        # Clean up the partial temp file if anything goes wrong
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e
        
    finally:
        # 6. Move the model back to its original GPU device
        model.to(original_device)    

def model_load(which_model="conv"):
    # Construct file name
    if which_model == "conv":
        prefix = "conv"
    else:
        prefix = "lin"
    suffix = "_model.pt"
    filename = f"{prefix}{suffix}"
    # Get the directory where this Python script is located
    script_dir = Path(__file__).resolve().parent
    # Join it with your file path
    file_path = script_dir / "Saved_models" / filename 

    # 1. Define the CPU device
    device = torch.device("cpu")

    # 2. Initialize your model architecture on the CPU
    if which_model == "conv":
        model = Network()
    else:
        model = NetworkSimple()
        
    # 3. Safely load the weights mapped to CPU
    state_dict = torch.load(
        file_path, 
        map_location=device, 
        weights_only=True
    )
    model.load_state_dict(state_dict)
    print("Model loaded from ", file_path)

    # 4. Ensure the model is on CPU and set to evaluation mode
    model.to(device)
    model.eval()

    # 5. Return the model so you can use it outside
    return model



############################################## tests
def test_training(training_data,validation_data,test_data):
    # Initialize the network (note: mini_batch_size is no longer hardcoded into the model)
    net = NetworkSimple()

    # Run training
    train_and_evaluate(
        model=net, 
        train_data=training_data, 
        val_data=validation_data, 
        test_data=test_data, 
        epochs=10, 
        mini_batch_size=200, 
        lr=0.1,
        w_decay=1e-6
    )

    save_weights_model_cpu_safely(net, fil_name="lin")

    print("Training test is over.")
    return 1

def test_loading(training_data,validation_data,test_data):
    # Load the model
    loaded_model = model_load(which_model="lin")

    # # Retrieve the full x tensor
    # x_component = test_data.tensors[0]

    # # Run evaluation/inference
    # with torch.no_grad():
    #     predictions = loaded_model(x_component)

    # Final Test evaluation
    test_loader = DataLoader(validation_data, batch_size=500)
    test_acc = evaluate(loaded_model, test_loader, torch.device("cpu"))
    print(f"Final Validation Accuracy: {test_acc:.2f}%")
    test_loader = DataLoader(test_data, batch_size=500)
    test_acc = evaluate(loaded_model, test_loader, torch.device("cpu"))
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    print("Loading test is over.")
    return 1


##################################################################################################################
#### MAIN
##################################################################################################################
# Define a main() function for what should run when executed directly
def main():
    # ### 1. we first train the convolution model

    # # Loading data
    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper_convolution()

    # # Initialize the network (note: mini_batch_size is no longer hardcoded into the model)
    # net = Network()

    # # Run training
    # train_and_evaluate(
    #     model=net, 
    #     train_data=training_data, 
    #     val_data=validation_data, 
    #     test_data=test_data, 
    #     epochs=25, 
    #     mini_batch_size=200, 
    #     lr=0.1,
    #     w_decay=1e-5
    # )

    # save_weights_model_cpu_safely(net, fil_name="conv")

    # print("Training convolution is over.")

    ### 2. we then train the linear model

    # Loading data
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper_linear()

    # Initialize the network (note: mini_batch_size is no longer hardcoded into the model)
    net = NetworkSimple()

    # Run training
    train_and_evaluate(
        model=net, 
        train_data=training_data, 
        val_data=validation_data, 
        test_data=test_data, 
        epochs=30, 
        mini_batch_size=100, 
        lr=0.5,
        w_decay=0.5e-5
    )

    save_weights_model_cpu_safely(net, fil_name="lin")

    print("Training linear is over.")

# Guard the entry point
if __name__ == "__main__":
    main()