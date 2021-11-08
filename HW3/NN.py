from torch import nn


class FeedForward(nn.Module):
    def __init__(self, inputN, outputN, hiddenNs=None):
        """
        :param: inputN = integer. Number of input nodes to our network
        :param: hiddenNs = list of integers. One element for each hidden layer. Each element tells
                       us the size of that layer
        :param: outputN = integer. Nuber of output nodes for our network
        """

        super(FeedForward, self).__init__()

        if hiddenNs is None:
            hiddenNs = []

        layer_sizes = [inputN] + hiddenNs + [outputN]  # form list of all layer sizes in the network

        self._layers = []  # initialise container for the layer objects
        self._methods = []  # initialise container
        for i in range(len(layer_sizes) - 1):  # build the architecture
            self._layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self._methods.append(nn.LeakyReLU)

    def forward(self, x):
        """
        Performs a forward pass of the network by for-looping through the layers in sequence, passing a vector
        x from one layer to the next until the output.

        :param: x = input vector to the network
        returns network output
        """
        for i in range(len(self._layers) - 1):
            x = self._layers[i](x)
            x = self._methods[i](x)
        return x
