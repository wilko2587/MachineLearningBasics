import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from NN import FeedForwardSoftmax, trainNN, generate_learning_curve
import data_transformations as dt
import validation_utils as vu
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler as MM


def read_mnist(file_name):
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            data_set.append([label, attribs])
    return (data_set)


def show_mnist(file_name, mode):
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % data_set[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0], end='')
        print(' ')


def read_insurability(file_name):
    count = 0
    data = []
    with open(file_name, 'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n', '')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls], [x1, x2, x3]])
            count = count + 1
    return (data)


def univariate_insurability():
    '''
    function to view the underlying relationships between hyperparameters and the outpus
    '''
    train = read_insurability('three_train.csv')

    hparams = ['Age', 'Exercise', 'Cigarettes']
    targets = {0: 'Good', 1: 'Neutral', 2: 'Bad'}  # maps the target onto custom labels

    for row in train:  # reformat the labelled data so y=2 --> [0,0,1] (to match the NN output)
        n = row[0].copy()[0]
        row[0] = [0, 0, 0]
        row[0][n] = 1.0

    dt.univariate(train, hparams, targets)
    return


def insurability_epochtest():
    '''
    see how the validation accuracy vs training accuracy varies over different epochs we train for
    '''

    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    train, valid, test = dt.scale01(train, [train, valid, test])  # normalise the data

    f1s = []  # holder to record the f1 score on validation set for different powers

    # initialise a new NN
    NN = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
    loss_func = nn.MSELoss()  # mean square error loss function
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    generate_learning_curve(train,valid,NN,loss_func,optimizer,
                            max_epoch=100000,
                            method='stochastic',
                            plot=True)
    return


def dataengineer_insurability():
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    train, valid, test = dt.scale01(train, [train, valid, test])  # normalise the data

    f1s = []  # holder to record the f1 score on validation set for different powers
    powers = np.arange(1, 3, 0.1)  # powers to experiment with
    for power in powers:
        # initialise a new NN
        NN = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
        loss_func = nn.MSELoss()  # mean square error loss function
        optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

        _train = dt.power_up(train, 2, power)
        _valid = dt.power_up(valid, 2, power)
        _test = dt.power_up(test, 2, power)

        train_loss = trainNN(_train, NN, loss_func, optimizer,
                             max_epoch=500000,
                             loss_target=0.15,
                             method='stochastic',
                             plot=False)  # train the NN on our data

        # test on the validation set
        valid_preds = NN.forward(dt.extract_hparams(_valid))
        cfm = vu.confusion_matrix(dt.extract_targets(_valid),
                                  dt.binary_to_labels(valid_preds),
                                  classes=[0, 1, 2])

        p, r, f1 = vu.precision_recall_F1(cfm, 2)
        f1s.append(f1)

    f = plt.figure()
    plt.plot(powers, f1s)
    plt.xlabel('power')
    plt.ylabel('f1 for class 2')
    plt.suptitle('F1 vs exponent of hyperparam "cigarettes"')
    return


def classify_insurability():
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    train, valid, test = dt.scale01(train, [train, valid, test])  # normalise the data

    # initialise a new NN
    NN = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
    loss_func = nn.MSELoss()  # mean square error loss function
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    train_loss = trainNN(train, NN, loss_func, optimizer,
                         max_epoch=500000,
                         loss_target=0.15,
                         method='stochastic',
                         plot=False)  # train the NN on our data

    # test on the validation set
    test_preds = NN.forward(dt.extract_hparams(test))
    cfm = vu.confusion_matrix(dt.extract_targets(test),
                              dt.binary_to_labels(test_preds),
                              classes=[0, 1, 2])

    print('--- confusion matrix ---')
    print(cfm)
    classes = {0: "good", 1: "neutral", 2: "bad"}
    for _class in classes:
        p, r, f1 = vu.precision_recall_F1(cfm, _class)
        print("class {} f1: {}".format(classes[_class], f1))
    return


def classify_mnist():
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    show_mnist('mnist_test.csv', 'pixels')

    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics


def classify_mnist_reg():
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    show_mnist('mnist_test.csv', 'pixels')

    # add a regularizer of your choice to classify_mnist()


def classify_insurability_manual():
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN


def main():
    # classify_insurability()
    univariate_insurability()
    insurability_epochtest()
    # classify_mnist()
    # classify_mnist_reg()
    # classify_insurability_manual()


if __name__ == "__main__":
    main()
