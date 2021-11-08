from torch import nn, tensor
import torch
import datetime

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

        self._layers = nn.ModuleList()  # initialise container for the layer objects
        self._methods = []  # initialise container
        for i in range(len(layer_sizes) - 1):  # build the architecture
            self._layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self._methods.append(nn.LeakyReLU())

    def forward(self, x):
        """
        Performs a forward pass of the network by for-looping through the layers in sequence, passing a vector
        x from one layer to the next until the output.

        :param: x = input vector to the network
        returns network output
        """
        #assert our vector x is a tensor
        x = tensor(x)
        for i in range(len(self._layers)): #iterate through the layers, passing x from one layer to the next
            x = self._layers[i](x)
            x = self._methods[i](x)
        return x


def trainNN(data, model, loss_func, optimizer):
    model.train()
    train_loss = []

    now = datetime.datetime.now()

    for batch, (y, X) in enumerate(data):

        # make predictions
        pred = model.forward(X)

        # make sure our datatypes are good
        pred = pred.to(torch.float32)
        y = tensor(y).to(torch.float32)

        # calculate the loss
        loss = loss_func(pred, y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0: #print out every 10 iterations...
            loss, current = loss.item(), batch * len(X)
            iters = 10 * len(X)
            then = datetime.datetime.now()
            iters /= (then - now).total_seconds()
            print(f"loss: {loss:>6f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
            now = then
            train_loss.append(loss)

    return train_loss