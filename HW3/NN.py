from torch import nn, tensor
import torch
import numpy as np

def softmax(x):
    '''
    returns softmax of x
    '''
    #x = np.array(x.detach().numpy()) #convert to number
    return torch.div(torch.exp(x),torch.sum(torch.exp(x),1).unsqueeze(1))


class FeedForwardSoftmax(nn.Module):
    def __init__(self, inputN, outputN, hiddenNs=None, bias=True):
        """
        Feed-forward linear neural network, using softmax at the output layer to normalise the outputs.

        :param: inputN = integer. Number of input nodes to our network
        :param: hiddenNs = list of integers. One element for each hidden layer. Each element tells
                       us the size of that layer
        :param: outputN = integer. Nuber of output nodes for our network
        :param: bias = True/False bool. If True pytorch will add a bias term into the calculations
        """

        super(FeedForwardSoftmax, self).__init__()

        if hiddenNs is None:
            hiddenNs = []

        layer_sizes = [inputN] + hiddenNs + [outputN]  # form list of all layer sizes in the network

        self._layers = nn.ModuleList()  # initialise container for the layer objects
        for i in range(len(layer_sizes) - 1):  # build the architecture
            self._layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias))

    def forward(self, x):
        """
        Performs a forward pass of the network by for-looping through the layers in sequence, passing a vector
        x from one layer to the next until the output.

        :param: x = input vector to the network
        returns network output
        """
        #assert our vector x is a tensor
        x = tensor(x).to(torch.float32)

        for i in range(len(self._layers)): #iterate through the layers, passing x from one layer to the next
            x = self._layers[i](x)
        return softmax(x)

    def get_weights(self):
        '''
        quick function to return the weights in the NN
        '''
        weights = []
        for layer in self._layers:
            weights.append(layer.weight)
        return weights


def trainNN(dataset, model, loss_func, optimizer, max_epoch = 50,
            loss_target = 0.1):
    '''
    :param: dataset = list holding data (in Demeter's format)
    :model: torch.nn.Module object -> our neural network object
    :loss_func: function to calculate the loss
    :optimizer: torch.optim object -> our optimizing function
    '''

    model.train() # tell the model we're training
    train_loss = []

    # set up the data
    X = tensor([d[1] for d in dataset])
    y = tensor([d[0] for d in dataset])

    # endusure the datatypes are okay
    X = X.to(torch.float32)
    y = y.to(torch.float32)

    loss = 1e8 # initialise to a value somewhere above the threshold
    epoch = 0 # counter for which training epoch we are in
    while loss > loss_target and epoch < max_epoch:
        # make predictions
        pred = model.forward(X)

        # calculate the loss
        loss = loss_func(pred, y)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        print('loss: ', loss)
        train_loss.append(loss)
        epoch += 1

    if loss <= loss_target:
        reason = "loss small enough!"
    else:
        reason = "max epoch reached ({})".format(max_epoch)

    print("Training complete! : {}".format(reason)) # print we're complete and reason the training stopped
    return train_loss
