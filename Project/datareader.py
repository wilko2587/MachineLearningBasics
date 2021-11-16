import pandas as pd
import dataclean_utils as dc
import torch


def read(filename, path='./data/'):
    '''
    basic function to read a csv file from path (default ./data/)
    '''
    csv = pd.read_csv(path + filename)
    return csv


def split_hyperparams_target(data, targetvar):
    '''
    returns the hyperparameters separated from the target inside data
    '''

    targetvar_cols = [col for col in data.columns if col[:len(targetvar)] == targetvar]
    target = data[targetvar_cols]
    hparams = data.drop(columns=targetvar_cols)
    return hparams, target


def read_cont():
    '''
    read the continuous data
    '''
    filename = "deid_full_data_cont.csv"
    data = read(filename)
    hparams, target = split_hyperparams_target(data, "stage")
    return (hparams, target)


def read_cat():
    '''
    read the categorical data
    '''
    filename = "deid_full_data_cat.csv"
    data = read(filename)
    hparams, target = split_hyperparams_target(data, "stage_category_int")
    hparams = dc.explode(hparams)
    return (hparams, target)


def generate_sets(data, splits = [70,15,15]):
    '''
    takes data, and splits into subsets
    :param: data = list of 2 dataframes. first dataframe is hyperparams, 2nd is target. must be same length
    :param splits = list of how much % of data we want in each subset
    :returns N lists of data, one for each split. each list contains 2 pd dataframes [hparams, target]
    '''

    assert(sum(splits)==100)

    split_data = []
    _prior = 0
    for split in splits:
        _n = _prior + int(len(data[0])*split/100.)
        subset = [torch.tensor(d.iloc[_prior:_n].values).to(torch.float32) for d in data]
        split_data.append(subset)
        _prior = _n

    return split_data
