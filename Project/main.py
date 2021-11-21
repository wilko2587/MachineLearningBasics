import feedforward as ff
import datareader as dr
import transformation_utils as tu
from torch import nn
import torch

catdata = dr.read_cat()

train, valid, test = dr.generate_sets(catdata, splits=[70, 15, 15])
train, valid, test = tu.scale01(train, [train, valid, test])

# lets make a simple feed forward NN with one hidden layer, softmax output
net = ff.FeedForwardSoftmax(len(train[0][0]), 3, hiddenNs=[3])  # N x 3 x 3 neural net
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

losses = ff.trainNN(train, net, loss, optimizer,
                    max_epoch=100000,
                    loss_target=0.72,
                    method='batch',  # pick "batch" or "stochastic"
                    plot=True,
                    verbosity=True,

                    # L1 and L2 regularisation strengths. NB: you can use a combination of both - this is called an
                    # elastic net
                    _lambdaL1=0.01,
                    _lambdaL2=0)

### because we have a lot of features, some that may not be relevant, i want to try L1 regularization
