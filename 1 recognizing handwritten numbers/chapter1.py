from coursecode.src import mnist_loader
from coursecode.src import network

training_data, validation_data, test_data = (
    mnist_loader.load_data_wrapper()
)

net = network.Network([784, 10])

net.SGD(training_data, 20, 10, 2.5, test_data=test_data)
