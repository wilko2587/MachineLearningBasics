import feedforward as ff
import datareader as dr
import transformation_utils as tu
import validation_utils as vu
from torch import nn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

def neural_net():
    '''
    NOTES
    -> because we have a lot of features, some that may not be relevant, i want to try L1 regularization
        L1 Functionality is now added into the ff.trainNN() function. I have put in L1 and L2 regularisation terms.
        Note that if we have too much time on our hands, we can use both of these in tandem/combination with eachother.
        This is a common method called an "elastic net", which gives a bit of the benefit of both L1 and L2 methods.

    -> For L1 regularisation in particular, we should progress by cross-validating the dataset into multiple parts
    to work out which parameters are useless with a higher degree of certainty. Some modules normally have this functionality
    built in, but I'm not sure about pytorch.
    '''

    catdata = dr.read_cat()

    train, valid, test = dr.generate_sets(catdata, splits=[70, 15, 15])
    train, valid, test = tu.scale01(train, [train, valid, test])

    # lets make a simple feed forward NN with one hidden layer, softmax output
    net = ff.FeedForwardSoftmax(len(train[0][0]), 3, hiddenNs=[20, 20])  # N x 3 x 3 neural net
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-5, momentum=0.)

    ff.generate_learning_curve(train, valid, net, loss, optimizer,
                               max_epoch = 100000,
                               method = 'batch',
                               minibatch_size=300,
                               outMethod=True,
                               _lambdaL1=0,
                               _lambdaL2=0)

    #losses = ff.trainNN(train, net, loss, optimizer,
    #                    max_epoch=100000,
    #                    loss_target=0.47,
    #                    method='minibatch',  # pick "batch" or "stochastic" or "minibatch"
    #                    minibatch_size=300,
    #                    plot=True,
    #                    verbosity=True,
    #
    #                    # L1 and L2 regularisation strengths. NB: you can use a combination of both - this is called an
    #                    # elastic net
    #                    _lambdaL1=1e-6,
    #                    _lambdaL2=0,
    #                    outMethod = True)

    test_predictions = tu.binary_to_labels(net.forward(valid[0], outMethod=True))
    test_true = valid[1].squeeze()

    cm = vu.confusion_matrix(test_true,test_predictions,classes = [0,1,2])
    F10 = vu.precision_recall_F1(cm,0)[2]
    F11 = vu.precision_recall_F1(cm,1)[2]
    F12 = vu.precision_recall_F1(cm,2)[2]

    print('--- confusion matrix ---')
    print(cm)

    print('F-measures \n class 0: {} \n class 1: {} \n class 2: {}'.format(F10,F11,F12))

    return None

def rf():
    x,Y = dr.read_cont() # load data

    # print(type(x),type(Y))

    labels = list()

    for each in Y.to_numpy():
        labels.append(each[0])

    exclude = ['id','bnp','a1c','chol'] # drop id because it isn't useful, values with high missingness
    x.drop(columns = exclude, axis=1, inplace=True)

    x['gender'].replace(to_replace=['Male', 'Female',], value=[0, 1], inplace=True) # make numeric
    smoke_cat = x['smoke'].unique()
    x['smoke'].replace(to_replace=smoke_cat,value=np.arange(11),inplace=True)
    final_col = x.columns

    # Missing values imputed as mean
    # Data scaled to mean = 0, var = 1

    simp_imp = SimpleImputer(strategy='mean').fit(x)
    x_imp = simp_imp.transform(x)

    scaler = StandardScaler().fit(x_imp)
    x_final = scaler.transform(x_imp)

    x = pd.DataFrame(x_final)
    x.columns = final_col

    rf = RandomForestClassifier(random_state = 0)
    rf.fit(x,labels)
    print(rf.score(x,labels)) # 99% accuracy!!! but I haven't split train/test :)
    # print(rf)



if __name__ == '__main__':
    # neural_net()
    rf() 