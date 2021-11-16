import feedforward as ff
import datareader as dr
import dataclean_utils as dc

data = dc.explode(dr.read('deid_full_data_cat.csv')) # our dataset as pandas dataframe

train = data.iloc[0:int(len(data)*0.70)]
valid = data.iloc[int(len(data)*0.70):int(len(data)*0.85)]
test = data.iloc[int(len(data)*0.85):]

#lets make a simple feed forward NN with one hidden layer, and softmax output to train and test the network

net = ff.FeedForwardSoftmax()