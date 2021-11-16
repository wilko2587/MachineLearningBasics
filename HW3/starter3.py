import copy
import sys
import random
import math
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from NN import FeedForwardSoftmax, trainNN, generate_learning_curve
import transformation_utils as tu
import validation_utils as vu
import matplotlib.pyplot as plt
from tqdm import tqdm
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

    tu.univariate(train, hparams, targets)
    return


def insurability_learningcurve():
    '''
    see how the validation accuracy vs training accuracy varies over different epochs we train for
    '''

    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    train, valid, test = tu.scale01(train, [train, valid, test])  # normalise the data

    train = tu.power_up(train,2,7.5) # raise Cigarettes to power of 6.5
    valid = tu.power_up(valid,2,7.5)

    # initialise a new NN
    NN = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    generate_learning_curve(train, valid, NN, loss_func, optimizer,
                            max_epoch=200000,
                            method='stochastic',
                            plot=True)
    return


def insurability_learningcurve_differentmoms():
    '''
    see how the training loss curve varies when we use two different learning parameters (ie: learning rate)
    '''

    train = read_insurability('three_train.csv')

    train = tu.scale01(train, [train])[0]  # normalise the data

    train = tu.power_up(train,2,6.5) # raise Cigarettes to power of 6.5

    # initialise a new NN
    NN1 = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
    NN2 = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
    loss_func = nn.CrossEntropyLoss()  # mean square error loss function
    optimizer1 = torch.optim.SGD(NN1.parameters(), lr=1e-3, momentum=0.9)
    optimizer2 = torch.optim.SGD(NN2.parameters(), lr=1e-3, momentum=0.7)

    training_curve1 = trainNN(train, NN1, loss_func, optimizer1,
                            max_epoch=300000,
                            loss_target=-1,
                            method='stochastic',
                            plot=False)

    training_curve2 = trainNN(train, NN2, loss_func, optimizer2,
                            max_epoch=300000,
                            loss_target = -1,
                            method='stochastic',
                            plot=False)

    f = plt.figure()
    plt.plot(training_curve1,label='momentum = 0.9')
    plt.plot(training_curve2,label='momentum = 0.7')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.suptitle("Loss curves for different momentums")
    plt.legend()
    return


def insurability_learningcurve_differentLR():
    '''
    see how the training loss curve varies when we use two different learning parameters (ie: learning rate)
    '''

    train = read_insurability('three_train.csv')

    train = tu.scale01(train, [train])[0]  # normalise the data

    train = tu.power_up(train,2,6.5) # raise Cigarettes to power of 6.5

    # initialise a new NN
    NN1 = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
    NN2 = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
    loss_func = nn.CrossEntropyLoss()  # mean square error loss function
    optimizer1 = torch.optim.SGD(NN1.parameters(), lr=1e-3, momentum=0.9)
    optimizer2 = torch.optim.SGD(NN2.parameters(), lr=3e-3, momentum=0.9)

    training_curve1 = trainNN(copy.deepcopy(train), NN1, loss_func, optimizer1,
                            max_epoch=100000,
                            loss_target=-1,
                            method='stochastic',
                            plot=False)

    training_curve2 = trainNN(copy.deepcopy(train), NN2, loss_func, optimizer2,
                            max_epoch=100000,
                            loss_target = -1,
                            method='stochastic',
                            plot=False)

    f = plt.figure()
    plt.plot(training_curve1,label='LR = 1e-3')
    plt.plot(training_curve2,label='LR = 3e-3')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.suptitle("Loss curves for different learning rates")
    plt.legend()
    return


def insurability_testbias():
    '''
    see how the validation accuracy varies when using bias and when not using bias.
    '''

    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    train, valid, test = tu.scale01(train, [train, valid, test])  # normalise the data

    #train = tu.power_up(train, 2, 6.5)  # raise cigarettes to power of 6.5
    #valid = tu.power_up(valid, 2, 6.5)

    results = pd.DataFrame(index=[True, False], columns=['Good', 'Neutral', 'Bad'])
    # initialise a new NN without bias
    print("Testing effect of bias...")
    for bias in [True, False]:
        NN = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=bias)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
        loss_func = nn.CrossEntropyLoss()  # mean square error loss function
        optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)
        trainNN(train, NN, loss_func, optimizer, max_epoch=100000, loss_target=0.08, method='stochastic', plot=False)

        # test on the validation set
        valid_preds = NN.forward(tu.extract_hparams(valid))
        cfm = vu.confusion_matrix(tu.extract_targets(valid),
                                  tu.binary_to_labels(valid_preds),
                                  classes=[0, 1, 2])
        classes = {0: "Good", 1: "Neutral", 2: "Bad"}
        for _class in classes:
            p, r, f1 = vu.precision_recall_F1(cfm, _class)
            results.loc[bias, classes[_class]] = f1

    results.index = ['bias', 'no bias']
    results = results.reset_index().melt(id_vars=['index']).rename(columns={'variable': 'classification',
                                                                            'value': 'F-Score',
                                                                            'index': 'bias'})

    sns.barplot(data=results, x='bias', y='F-Score', hue='classification')
    plt.suptitle("Effect of Bias")
    return


def insurability_testmomentum():
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    train, valid, test = tu.scale01(train, [train, valid, test])  # normalise the data

    train = tu.power_up(train,2,6.5) #power cigarettes by 6.5
    valid = tu.power_up(valid,2,6.5)

    momentums = np.arange(0.5, 1, 0.05)  # momentums to experiment with

    f1s = pd.DataFrame(data=0,
                       index=momentums,
                       columns=["good", "neutral",
                                "bad"])  # holder to record the f1 score on validation set for different powers

    for mom in momentums: # run a few times and average to smooth out
        fs = {0:[],1:[],2:[]}
        for n in range(5):
            # initialise a new nn
            NN = FeedForwardSoftmax(3, 3, hiddenNs = [2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
            loss_func = nn.CrossEntropyLoss()  # mean square error loss function
            optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=mom)

            train_loss = trainNN(train, NN, loss_func, optimizer,
                                 max_epoch=500000,
                                 loss_target=0.75,
                                 method='stochastic',
                                 plot=False,
                                 verbosity=True)  # train the nn on our data

            # test on the validation set
            valid_preds = NN.forward(tu.extract_hparams(valid))
            cfm = vu.confusion_matrix(tu.extract_targets(valid),
                                      tu.binary_to_labels(valid_preds),
                                      classes=[0, 1, 2])
            classes = {0: "good", 1: "neutral", 2: "bad"}
            for _class in classes:
                p, r, f1 = vu.precision_recall_F1(cfm, _class)
                fs[_class].append(f1)

        for _class in classes:
            f1s.loc[mom, classes[_class]] = sum(fs[_class])/len(fs[_class])

    f1s['Mean F-Score'] = f1s.mean(axis=1)

    f1s = f1s.reset_index().melt(id_vars=['index']).rename(columns={"index": "momentum",
                                                                    "value": "f-score",
                                                                    "variable": "classification"})
    sns.lineplot(data=f1s, x='momentum', y='f-score', hue='classification')
    plt.suptitle("effect of momentum on performance")
    return


def insurability_testlosstarget():
    '''
    duplicate of insurabilty_testmomentum(), but here we vary the loss target of the model.
    '''

    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    train, valid, test = tu.scale01(train, [train, valid, test])  # normalise the data

    train = tu.power_up(train,2,6.5)
    valid = tu.power_up(valid,2,6.5)

    losstargets = np.arange(0.70, 1.0, 0.02)

    f1s = pd.DataFrame(data=0,
                       index=losstargets,
                       columns=["Good", "Neutral",
                                "Bad"])  # holder to record the f1 score on validation set for different powers

    for loss in losstargets:
        # initialise a new NN
        NN = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
        loss_func = nn.CrossEntropyLoss()  # mean square error loss function
        optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

        train_loss = trainNN(train, NN, loss_func, optimizer,
                             max_epoch=500000,
                             loss_target=loss,
                             method='stochastic',
                             plot=False,
                             verbosity=True)  # train the NN on our data

        # test on the validation set
        valid_preds = NN.forward(tu.extract_hparams(valid))
        cfm = vu.confusion_matrix(tu.extract_targets(valid),
                                  tu.binary_to_labels(valid_preds),
                                  classes=[0, 1, 2])
        classes = {0: "Good", 1: "Neutral", 2: "Bad"}
        for _class in classes:
            p, r, f1 = vu.precision_recall_F1(cfm, _class)
            f1s.loc[loss, classes[_class]] = f1

    f1s = f1s.reset_index().melt(id_vars=['index']).rename(columns={"index": "Loss Target",
                                                                    "value": "F-Score",
                                                                    "variable": "Classification"})
    sns.lineplot(data=f1s, x='Loss Target', y='F-Score', hue='Classification')
    plt.suptitle("Effect of Training Loss on Performance")
    return


def insurability_testpowers():
    '''
    duplicate of insurabilty_testmomentum(), but here we vary raising the hyperparam "cigarettes" to a power
    '''

    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    train, valid, test = tu.scale01(train, [train, valid, test])  # normalise the data

    powers = np.arange(1, 10, 0.5)

    f1s = pd.DataFrame(data=0,
                       index=powers,
                       columns=["Good", "Neutral",
                                "Bad"])  # holder to record the f1 score on validation set for different powers

    for pow in powers:
        fs = {0:[],1:[],2:[]}
        print('power: ',pow)
        for n in range(0,5):
            # initialise a new NN
            NN = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
            loss_func = nn.CrossEntropyLoss()  # mean square error loss function
            optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

            _train = tu.power_up(train, 2, pow) #raise the 2nd column (cigarettes) to a power
            _valid = tu.power_up(valid, 2, pow)

            train_loss = trainNN(_train, NN, loss_func, optimizer,
                                 max_epoch=500000,
                                 loss_target=0.67,
                                 method='stochastic',
                                 plot=False,
                                 verbosity=True)  # train the NN on our data

            # test on the validation set
            valid_preds = NN.forward(tu.extract_hparams(_valid))
            cfm = vu.confusion_matrix(tu.extract_targets(_valid),
                                      tu.binary_to_labels(valid_preds),
                                      classes=[0, 1, 2])
            classes = {0: "Good", 1: "Neutral", 2: "Bad"}
            for _class in classes:
                p, r, f1 = vu.precision_recall_F1(cfm, _class)
                fs[_class].append(f1)

        for _class in classes:
            f1s.loc[pow, classes[_class]] = sum(fs[_class])/len(fs[_class])

    f1s['Mean F-Score'] = f1s.mean(axis=1)
    f1s = f1s.reset_index().melt(id_vars=['index']).rename(columns={"index": "Exponent",
                                                                    "value": "F-Score",
                                                                    "variable": "Classification"})
    sns.lineplot(data=f1s, x='Exponent', y='F-Score', hue='Classification')
    plt.suptitle("Effect of Cigarette exponent on Performance")
    return


def classify_insurability():
    # put our final model here
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    train, valid, test = tu.scale01(train, [train, valid, test])  # normalise the data

    train = tu.power_up(train,2,7.5)
    test = tu.power_up(test,2,7.5)

    # initialise a new NN
    NN = FeedForwardSoftmax(3, 3, hiddenNs=[2], bias=True)  # 3 inputs, 2 hidden, 3 outputs as per slides 11-8.
    loss_func = nn.CrossEntropyLoss()  # mean square error loss function
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    train_loss = trainNN(train, NN, loss_func, optimizer,
                         max_epoch=500000,
                         loss_target=0.65,
                         method='stochastic',
                         plot=False)  # train the NN on our data

    # test on the validation set
    test_preds = NN.forward(tu.extract_hparams(test))
    cfm = vu.confusion_matrix(tu.extract_targets(test),
                              tu.binary_to_labels(test_preds),
                              classes=[0, 1, 2])

    print('--- confusion matrix ---')
    print(cfm)
    classes = {0: "good", 1: "neutral", 2: "bad"}
    for _class in classes:
        p, r, f1 = vu.precision_recall_F1(cfm, _class)
        print("class {} f1: {}".format(classes[_class], f1))
    return


def learningcurve_mnist():
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    #show_mnist('mnist_test.csv', 'pixels')

    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    # with mnist, we need to turn the data from strings into integers
    train, valid, test = tu.str2int([train, valid, test])

    train_lowD, test_lowD = tu.low_variance_filter(train, test, 60)

    # turn data to binary
    train = tu.data_bin(train)
    valid = tu.data_bin(valid)
    test = tu.data_bin(test)

    # run data through variance filter

    # initialise a new NN
    NN = FeedForwardSoftmax(len(train[0][1]), 10, hiddenNs=[10], bias=True)
    loss_func = nn.CrossEntropyLoss()  # mean square error loss function
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    train_loss = trainNN(train, NN, loss_func, optimizer,
                         max_epoch=500000,
                         loss_target=1.30,
                         method='batch',
                         plot=False)  # train the NN on our data

    # test on the validation set
    test_preds = NN.forward(tu.extract_hparams(test))
    cfm = vu.confusion_matrix(tu.extract_targets(test),
                              tu.binary_to_labels(test_preds),
                              classes=range(0,10))

    print('--- confusion matrix ---')
    print(cfm)
    classes = range(0,10)
    for _class in classes:
        p, r, f1 = vu.precision_recall_F1(cfm, _class)
        print("class {} f1: {}".format(_class, f1))
    return


def mnist_testvarthresh():
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    #show_mnist('mnist_test.csv', 'pixels')

    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    # with mnist, we need to turn the data from strings into integers
    train, valid, test = tu.str2int([train, valid, test])

    # turn data to binary
    train = tu.data_bin(train)
    valid = tu.data_bin(valid)
    test = tu.data_bin(test)

    # run data through variance filter
    varthreshes = range(0,90,5)

    f1s = pd.DataFrame(data=0,
                       index=varthreshes,
                       columns=range(0,10))

    for thresh in varthreshes:
        print('-------')
        print(thresh)
        train_lowD, valid_lowD = tu.low_variance_filter(train, valid, thresh)

        # initialise a new NN
        NN = FeedForwardSoftmax(len(train[0][1]), 10, hiddenNs=[10], bias=True)
        loss_func = nn.CrossEntropyLoss()  # mean square error loss function
        optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

        losscurve = trainNN(train, NN, loss_func, optimizer,
                             loss_target = 1.60,
                             max_epoch=300000,
                             method='stochastic',
                             plot=False)  # train the NN on our data

        valid_preds = NN.forward(tu.extract_hparams(valid)) #generate predictions

        #generate confusion matrix
        cfm = vu.confusion_matrix(tu.extract_targets(valid),
                                  tu.binary_to_labels(valid_preds),
                                  classes=range(10))

        classes = range(10)
        for _class in classes:
            p, r, f1 = vu.precision_recall_F1(cfm, _class)
            f1s.loc[thresh, _class] = f1

    f1s['Mean F-Score'] = f1s.mean(axis=1)
    #f1s = f1s.reset_index().melt(id_vars=['index']).rename(columns={"index": "Variance Threshold",
    #                                                                "value": "F-Score",
    #                                                                "variable": "Classification"})

    fig2 = plt.figure()
    ax2 = plt.subplot(111)
    sns.lineplot(data=f1s['Mean F-Score'])
    plt.suptitle("Effect of variance threshold filter on performance")
    return


def classify_mnist():
    train = read_mnist('mnist_train.csv')
    test = read_mnist('mnist_test.csv')
    #show_mnist('mnist_test.csv', 'pixels')

    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    # with mnist, we need to turn the data from strings into integers
    train, test = tu.str2int([train, test])

    # turn data to binary
    train = tu.data_bin(train)
    test = tu.data_bin(test)

    train, test = tu.low_variance_filter(train, test, 50)

    # initialise a new NN
    NN = FeedForwardSoftmax(len(train[0][1]), 10, hiddenNs=[10], bias=True)
    loss_func = nn.CrossEntropyLoss()  # mean square error loss function
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    losscurve = trainNN(train, NN, loss_func, optimizer,
                         loss_target = 1.50,
                         max_epoch=300000,
                         method='stochastic',
                         plot=False)  # train the NN on our data

    test_preds = NN.forward(tu.extract_hparams(test)) #generate predictions

    #generate confusion matrix
    cfm = vu.confusion_matrix(tu.extract_targets(test),
                              tu.binary_to_labels(test_preds),
                              classes=range(10))

    print('---- confusion matrix ----')
    print(cfm)
    classes = range(0,10)
    for _class in classes:
        p, r, f1 = vu.precision_recall_F1(cfm, _class)
        print("class {} f1: {}".format(_class, f1))
    return



def mnist_learningcurve():
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    #show_mnist('mnist_test.csv', 'pixels')

    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    # with mnist, we need to turn the data from strings into integers
    train, valid, test = tu.str2int([train, valid, test])

    # turn data to binary
    train = tu.data_bin(train)
    valid = tu.data_bin(valid)
    test = tu.data_bin(test)

    # initialise a new NN
    NN = FeedForwardSoftmax(len(train[0][1]), 10, hiddenNs=[10], bias=True)
    loss_func = nn.CrossEntropyLoss()  # mean square error loss function
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    generate_learning_curve(train, valid, NN, loss_func, optimizer,
                         max_epoch=100000,
                         method='stochastic',
                         plot=True)  # train the NN on our data

    return


def mnist_learningcurveL2():
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    #show_mnist('mnist_test.csv', 'pixels')

    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    # with mnist, we need to turn the data from strings into integers
    train, valid, test = tu.str2int([train, valid, test])

    # turn data to binary
    train = tu.data_bin(train)
    valid = tu.data_bin(valid)
    test = tu.data_bin(test)

    # initialise a new NN
    NN = FeedForwardSoftmax(len(train[0][1]), 10, hiddenNs=[10], bias=True)
    loss_func = nn.CrossEntropyLoss()  # mean square error loss function
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    generate_learning_curve(train, valid, NN, loss_func, optimizer,
                         max_epoch=100000,
                         method='stochastic',
                         plot=True,
                         _lambda=2e-3)  # train the NN on our data

    return



def mnist_testL2():
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')

    #show_mnist('mnist_test.csv', 'pixels')

    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    # with mnist, we need to turn the data from strings into integers
    train, valid = tu.str2int([train, valid])

    # turn data to binary
    train = tu.data_bin(train)
    valid = tu.data_bin(valid)

    # run data through variance filter
    alphas = [0,1e-5,2e-5,5e-5,7e-5,1e-4,2e-4,5e-4,7e-4,1e-3,2e-3,5e-3,7e-3,1e-2]

    f1s = pd.DataFrame(data=0,
                       index=alphas,
                       columns=range(0,10))

    for a in alphas:
        print('a: ',a)
        # initialise a new NN
        NN = FeedForwardSoftmax(len(train[0][1]), 10, hiddenNs=[10], bias=True)
        loss_func = nn.CrossEntropyLoss()  # mean square error loss function
        optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

        losscurve = trainNN(train, NN, loss_func, optimizer,
                             loss_target = 1.60,
                             max_epoch=500000,
                             method='stochastic',
                             plot=False,
                             _lambda = a)  # train the NN on our data

        valid_preds = NN.forward(tu.extract_hparams(valid)) #generate predictions

        #generate confusion matrix
        cfm = vu.confusion_matrix(tu.extract_targets(valid),
                                  tu.binary_to_labels(valid_preds),
                                  classes=range(10))

        classes = range(10)
        for _class in classes:
            p, r, f1 = vu.precision_recall_F1(cfm, _class)
            f1s.loc[a, _class] = f1

    f1s['Mean F-Score'] = f1s.mean(axis=1)
    f1s_2 = f1s.reset_index().melt(id_vars=['index']).rename(columns={"index": "Exponent",
                                                                    "value": "F-Score",
                                                                    "variable": "Classification"})
    fig1 = plt.figure()
    ax1 = plt.subplot(111)
    g = sns.lineplot(data=f1s['Mean F-Score'], ax = ax1)
    g.set(xscale='log')
    plt.xlabel('Lambda')
    plt.ylabel("Mean F-Score")
    plt.suptitle("Effect of regularisation constant on performance")
    #fig2 = plt.figure()
    #ax2 = plt.subplot(111)
    #sns.lineplot(data=f1s_2, x='Threshold', y='F-Score', hue='Classification',ax = ax2)
    #plt.suptitle("Effect of regularisation strength on performance")
    return


def mnist_learningcurvesL2_comparison():
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    #show_mnist('mnist_test.csv', 'pixels')

    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    # with mnist, we need to turn the data from strings into integers
    train, valid = tu.str2int([train, valid])

    # turn data to binary
    train = tu.data_bin(train)

    train, valid = tu.low_variance_filter(train, valid, 50)

    # initialise first NN
    NN = FeedForwardSoftmax(len(train[0][1]), 10, hiddenNs=[10], bias=True)
    loss_func = nn.CrossEntropyLoss()  # mean square error loss function
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    losscurve1 = trainNN(train, NN, loss_func, optimizer,
                         loss_target = -1,
                         max_epoch=200000,
                         method='stochastic',
                         plot=False)  # train the NN on our data

    NN = FeedForwardSoftmax(len(train[0][1]), 10, hiddenNs=[10], bias=True)
    loss_func = nn.CrossEntropyLoss()  # mean square error loss function
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    losscurve2 = trainNN(train, NN, loss_func, optimizer,
                         loss_target = -1,
                         max_epoch=200000,
                         method='stochastic',
                         plot=False)  # train the NN on our data

    f = plt.figure()
    plt.plot(losscurve1, label="No Regularisation")
    plt.plot(losscurve2, label="Lambda = 2e-3")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.suptitle("Learning curves with/without regularisation")
    return


def classify_mnist_reg():
    # add a regularizer of your choice to classify_mnist()
    ## NB: REGULARISER BUILT INTO our trainNN() function

    train = read_mnist('mnist_train.csv')
    test = read_mnist('mnist_test.csv')

    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics

    # with mnist, we need to turn the data from strings into integers
    train, test = tu.str2int([train, test])

    # turn data to binary
    train = tu.data_bin(train)
    test = tu.data_bin(test)

    train, test = tu.low_variance_filter(train, test, 50)

    # initialise a new NN
    NN = FeedForwardSoftmax(len(train[0][1]), 10, hiddenNs=[10], bias=True)
    loss_func = nn.CrossEntropyLoss()  # mean square error loss function
    optimizer = torch.optim.SGD(NN.parameters(), lr=1e-3, momentum=0.9)

    losscurve = trainNN(train, NN, loss_func, optimizer,
                         loss_target = 1.50,
                         max_epoch=300000,
                         method='stochastic',
                         plot=False,
                         _lambda=2e-3)  # train the NN on our data

    test_preds = NN.forward(tu.extract_hparams(test)) #generate predictions

    #generate confusion matrix
    cfm = vu.confusion_matrix(tu.extract_targets(test),
                              tu.binary_to_labels(test_preds),
                              classes=range(10))

    print('---- confusion matrix ----')
    print(cfm)
    classes = range(0,10)
    for _class in classes:
        p, r, f1 = vu.precision_recall_F1(cfm, _class)
        print("class {} f1: {}".format(_class, f1))
    return


def classify_insurability_manual():
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN


def main():
    # classify_insurability()
    # univariate_insurability()
    # insurability_learningcurve()
    # insurability_testbias()
    # insurability_testmomentum()
    # insurability_learningcurve_differentLR()
    # insurability_testlosstarget()
    # insurability_testpowers()
    # mnist_learningcurve()
    # mnist_learningcurveL2()
    classify_mnist_reg()
    # classify_insurability_manual()
    # mnist_testvarthresh()
    # classify_mnist()
    # mnist_learningcurvesL2_comparison()
    # mnist_testL2()


if __name__ == "__main__":
    main()
