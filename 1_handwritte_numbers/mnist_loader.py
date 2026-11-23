"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
from pathlib import Path
import pickle
import gzip

# Third-party libraries
import numpy as np
import torch
from torch.utils.data import TensorDataset

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    # Get the directory where this Python script is located
    script_dir = Path(__file__).resolve().parent

    # Join it with your file path
    file_path = script_dir / "Data/mnist.pkl.gz"
    # Or if it's in a subfolder: script_dir / "subfolder" / "your_file.pkl"
    f = gzip.open(file_path, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

# for the convolution network we wrap the data in a different way
def load_data_wrapper_convolution():
    tr_d, va_d, te_d = load_data()

    def to_dataset(data_tuple):
        # Reshape (N, 784) -> (N, 1, 28, 28) and convert to float32 tensor
        x = torch.tensor(data_tuple[0], dtype=torch.float32).reshape(-1, 1, 28, 28)
        # Keep labels as 1D integer tensors (N,) of type long
        y = torch.tensor(data_tuple[1], dtype=torch.long)
        return TensorDataset(x, y)

    training_data = to_dataset(tr_d)
    validation_data = to_dataset(va_d)
    test_data = to_dataset(te_d)

    return training_data, validation_data, test_data

# for the linear network we wrap the data in a different way
def load_data_wrapper_linear():
    tr_d, va_d, te_d = load_data()

    def to_dataset(data_tuple):
        # Reshape (N, 784) -> (N, 1, 28, 28) and convert to float32 tensor
        x = torch.tensor(data_tuple[0], dtype=torch.float32) 
        # Keep labels as 1D integer tensors (N,) of type long
        y = torch.tensor(data_tuple[1], dtype=torch.long)
        return TensorDataset(x, y)

    training_data = to_dataset(tr_d)
    validation_data = to_dataset(va_d)
    test_data = to_dataset(te_d)

    return training_data, validation_data, test_data