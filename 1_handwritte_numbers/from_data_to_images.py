"""
image_generator
~~~~~~~~~~~~

A library to generate a number image from the MNIST data.   
"""


# library defined by Michael Nielsen and adapted by me
import mnist_loader

# More standard librearies
import matplotlib.pyplot as plt
from pathlib import Path

#### simply create a 28x28 image from an (784,) array where each element is the graylevel of each pixel (and save the image with the name picture_name in png format)
def figure_from_array(arr, picture_name="pic.png"):

    # Reshape to 28x28 and use 'gray_r' (reversed grayscale) so 1 = black and 0 = white
    plt.imshow(arr.reshape((28, 28)), cmap="gray_r", vmin=0, vmax=1)
    plt.axis("off")  # Hides the axis ticks

    # Get the directory where this Python script is located
    script_dir = Path(__file__).resolve().parent

    # Join it with your file path
    file_path = script_dir / "Generated_images" / picture_name 
    #  Save the plot to a file (must be called BEFORE plt.show())
    plt.savefig(file_path, dpi=28, bbox_inches="tight")

    plt.show()
    return 1

############################################## tests
# training_data, validation_data, test_data = mnist_loader.load_data()

# print(training_data[1])

# arr = training_data[0][1000]

# figure_from_array(arr)
