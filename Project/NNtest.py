import feedforward as ff
import datareader as dr
import transformation_utils as tu
import validation_utils as vu
from torch import nn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


def neural_net():
    '''
    run a NN on a dummy dataset where performance is expected to be 100%. Used to debug the NN code.
    '''
    data = dr.read_linsep()

    train, valid, test = dr.generate_sets(data, splits=[70, 15, 15])
    train, valid, test = tu.scale01(train, [train, valid, test])

    # lets make a simple feed forward NN with one hidden layer, softmax output
    net = ff.FeedForwardSoftmax(len(train[0][0]), 2, hiddenNs=[5])  # N x 3 x 3 neural net
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)


    losses = ff.trainNN(train, net, loss, optimizer,
                        max_epoch=200000,
                        loss_target=0.10,
                        method='minibatch',  # pick "batch" or "stochastic" or "minibatch"
                        minibatch_size=300,
                        plot=True,
                        verbosity=True,

                        # L1 and L2 regularisation strengths. NB: you can use a combination of both - this is called an
                        # elastic net
                        _lambdaL1=0.,
                        _lambdaL2=0.,
                        outMethod = False)

    test_predictions = tu.binary_to_labels(net.forward(valid[0], outMethod=True))
    test_true = valid[1].squeeze()

    cm = vu.confusion_matrix(test_true,test_predictions,classes = [0,1])
    F10 = vu.precision_recall_F1(cm,0)[2]
    F11 = vu.precision_recall_F1(cm,1)[2]

    print('--- confusion matrix ---')
    print(cm)

    print('F-measures \n Predicted 0: {} \n Predicted 1: {}'.format(F10,F11))

    return None


if __name__ == '__main__':
    neural_net()