import seaborn as sns
import math_utils as mu
import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

sns.set_theme(style="darkgrid")

def univariate(data,hparams, targets):
    '''
    Function to plot the univariate relationships.
    ie: looking at individual hypermareters at a time.
    Y can take 3 values (1,0,0; 0,1,0; 0,0,1).
    Vary the hyperparameter, and see how each Y changes to view the underlying relationships.

    :param: data = data we're analysing, in demeter's format
    :hparams: names of the hyperparameters in 'data'
    :targets: dictionary mapping the target onto a label
    '''

    #split out the hyperparams (X) from the targets (y)
    X = pd.DataFrame([d[1] for d in data])
    X.columns = hparams
    y = pd.DataFrame([d[0] for d in data])
    y.columns = [targets[i] for i in y.columns] #name the target variables

    ylabel = pd.DataFrame([d[0].index(1) for d in data]) #get the true label of y (0,1,2 instead of [1,0,0];[0,1,0];[0,0,1])
    ylabel.columns = ['ylabel']

    axs = plt.subplots(1,len(X.columns))
    axs = {X.columns[i]:axs[1][i] for i in range(len(X.columns))} #nasty way of mapping axes to axes for later use
    for hparam in X.columns: #for each hyperparmeter lets look at the relationship to the target y
        ax = axs[hparam]
        plotdata = pd.DataFrame(X[hparam])

        # bin the hyperparam
        bins = np.linspace(X[hparam].min(), X[hparam].max(),20) #create 20 bins
        plotdata['binned'] = pd.cut(X[hparam], bins,labels=bins[:-1])

        # concatenate to leave us with dataframe with first column being a hyperparameter, the rest of the columns
        #   being y
        concat = pd.concat([plotdata,y,ylabel],axis=1)

        for i in y.columns:
            sns.lineplot(data = concat,
                         x = 'binned',
                         y = i,
                         estimator = np.mean,
                         ax= ax,
                         label=i)
        ax.set_xlabel('hyperparam: '+str(hparam))
        ax.set_ylabel('Observed Probability')
        ax.set_title(hparam)
        ax.legend()


def stepify(data, column, threshold):
    '''
    function to turn a column in data into a step-function, parameterised by "threshold"

    :param: data = the dataset (in demeter's format) we want to transform. List dtype
    :column: integer representing the index of the hyperparameter we want to edit
    :threshold: where the column is above threshold, we set to 1. Where below, we set to 0
    :returns: transformed data (same format as demeter's examples), with the column turned into step fn. List dtype
    '''
    for row in data:
        row[1][column] = int(row[1][column] >= threshold)
    return data #that was easy...


def power_up(data, column, power):
    '''
    function to raise a column in data to a power.
    --> If power is above 1, this should increase the impact of that hyperparameter being high in value.
    --> If power is below 1, this should increase the impact of that hyperparameter being low in value.

    :param: data = the dataset (in demeter's format) we want to transform. List dtype
    :column: integer representing the index of the hyperparameter we want to edit
    :power: the power we want to raise the column to
    :returns: transformed data (same format as demeter's examples), List dtype
    '''
    new_data = []
    datacopy = copy.deepcopy(data)
    for row in datacopy:
        new_data.append(row)
        new_data[-1][1][column] = new_data[-1][1][column]**power
    return new_data


def Zscore(train, items_to_norm):
    '''
    function to use the standard deviation and mean from columns in training data to normalise the test data

    :param: train = list of hyperparameters. axis0 are different examples. axis1 are different hyperparameters
    :param: items_to_norm = list of datas to normalise. each element has same format as train
            (can be different axis0 size, but same axis1 size)
    :returns: Normalised test data
    '''
    Nparams = len(train[0][1])

    for i in range(Nparams):
        _train = [x[1][i] for x in train]
        mean = np.mean(_train)
        std = np.std(_train)

        for item in items_to_norm:
            for row in item:
                row[1][i] = (row[1][i] - mean)/std

    return items_to_norm

def scale01(train, items_to_norm):
    '''
    function to scale the data in items_to_norm so it sits between 0->1 (ie: no negatives)
    '''

    Nparams = len(train[0][1])
    for i in range(Nparams):
        _train = [x[1][i] for x in train]
        max = np.max(_train)
        min = np.min(_train)
        for item in items_to_norm:
            for row in item:
                row[1][i] = (row[1][i] - min)/(max-min)
    return items_to_norm


def labels_to_binary(ys, Noutputs):
    '''
    converts y = 2 into [0,0,1] etc (to match NN output). Number of items in list should be given by number of output
    nodes in the NN in input argument to this function
    '''
    new_ys = []
    ys = ys.tolist()
    for row in ys:  # reformat the labelled data so y=2 --> [0,0,1] (to match the NN output)
        n = int(row)
        row = [0]*Noutputs
        row[n] = 1
        new_ys.append(row)
    new_ys = torch.tensor(new_ys)
    new_ys = new_ys.to(torch.float32)
    return new_ys


def binary_to_labels(ys):
    '''
    converts y = [0,0,1] into 2 etc (to match labels output) for insurability data
    '''
    return torch.argmax(ys,1)


def extract_hparams(data):
    '''
    function to extract the hyperparameters from demeter's data format
    '''
    hparams = []
    for row in data:
        hparams.append(row[1])
    hparams = torch.tensor(hparams)
    hparams = hparams.to(torch.float32)
    return hparams


def extract_targets(data):
    '''
    function to extract the targets from demeter's data format
    '''
    targets = []
    for row in data:
        targets.append(row[0][0])
    targets = torch.tensor(targets)
    targets = targets.to(torch.float32)
    return targets


def data_bin(data):
    """
    function to turn data from continuous into binary
    nb: "datas" structure is
    data[0] = a string label,
    data[1] = list of numbers we will turn to binary
    returns data in the format [data[0],new_data[1]]
    """
    lin_num = 0
    for line in data: # line[0] = label, line[1] = list with all values
        pix_num = 0
        for each in line[1]: # go into each individual value
            if int(each) > 0: # if greater than 1, replace that specific location with 1
                #data[data.index(line)][1][line[1].index(each)] = 1.0
                data[lin_num][1][pix_num] = 1
            else:
                data[lin_num][1][pix_num] = 0
            pix_num += 1
        lin_num += 1
    return data


def low_variance_filter(train_data, validation_data, pctile_thresh):
    '''
    Dimensionality reduction technique.
    We scan through each corresponding "pixel" between the different datas, and calculate that pixel's variance across
    the dataset.
    We then eliminate pixels that have a variance below "threshold"
    "pctile" defines the percentile of variance below which we cut
    "validation_data" is a list of other data (ie: validation/test data we want to also affect the changes on)
    '''

    pixel_variance = []  # initialise. All values will end up being overwritten
    npixels = len(train_data[0][1])
    for x in range(
            npixels):  # scan through each pixel, and work out what the variance is for that same pixel, accross the dataset.
        pixel_data = [data[1][x] for data in train_data]
        _var = mu.var(pixel_data)
        pixel_variance.append(_var)  # save the result in pixel_variance

    pixels_to_keep = [x for x in range(npixels) if
                      mu.pctile(pixel_variance[x], list(set(pixel_variance))) >= pctile_thresh]

    reduced_train = [[data[0], [data[1][x] for x in pixels_to_keep]] for data in train_data]
    reduced_valid = [[data[0], [data[1][x] for x in pixels_to_keep]] for data in validation_data]
    print('--> {} dimensions in input; {} dimensions in output'.format(len(train_data[0][1]),len(reduced_train[0][1])))
    return reduced_train, reduced_valid


def str2int(list_of_data):
    '''
    function to convert all data in list_of_data (which is a list of [train, valid, test] for example,
    from strings into integers
    Also puts the target in its own list, so a row in mnist data goes from: [7, [0,0,0,...]] -> [ [7], [0,0,0 ....]]
    which matches the insurability data format
    '''
    list_out = []
    for data in list_of_data:
        for row in data:
            row[0] = [int(row[0])] #first convert the target
            row[1] =  [int(i) for i in row[1]] #second convert the hparams
        list_out.append(data)
    return list_out

