import pandas as pd
import dataclean_utils as dc


def read(filename, path='./data/'):
    '''
    basic function to read a csv file from path (default ./data/)
    '''
    csv = pd.read_csv(path + filename)
    return csv


def get_hyperparams(data):
    '''
    returns the hyperparameters inside data
    '''

    return


f1 = read('deid_full_data_cat.csv')

f1a = dc.explode(f1)