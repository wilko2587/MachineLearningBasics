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
    net = ff.FeedForwardSoftmax(len(train[0][0]), 3, hiddenNs=[10])  # N x 3 x 3 neural net
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-5, momentum=0.)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

    #ff.generate_learning_curve(train, valid, net, loss, optimizer,
    #                           max_epoch = 100000,
    #                           method = 'batch',
    #                           minibatch_size=300,
    #                           outMethod=True,
    #                           _lambdaL1=0,
    #                           _lambdaL2=0)

    losses = ff.trainNN(train, net, loss, optimizer,
                        max_epoch=200000,
                        loss_target=0.2,
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

    cm = vu.confusion_matrix(test_true,test_predictions,classes = [0,1,2])
    F10 = vu.precision_recall_F1(cm,0)[2]
    F11 = vu.precision_recall_F1(cm,1)[2]
    F12 = vu.precision_recall_F1(cm,2)[2]

    print('--- confusion matrix ---')
    print(cm)

    print('F-measures \n class 0: {} \n class 1: {} \n class 2: {}'.format(F10,F11,F12))

    return None

def models():
    '''
    Beginning is all data prep.
    After this, data stored as (train_df, test_df) and labels as (train_labels, test_labels)
    '''

    data = dr.read('deid_full_data_cont.csv')

    data.drop(columns='id',axis=1,inplace=True) # do not need id
    data['gender'].replace(to_replace=['Male', 'Female',], value=[0, 1], inplace=True) # make gender/smoke numeric
    data['bnp'].replace(to_replace=['a'], value=['nan'],inplace=True) #typo in BNP data somewhere
    smoke_cat = data['smoke'].unique()
    data['smoke'].replace(to_replace=smoke_cat, value=np.arange(11), inplace=True)

    # exclude = ['bnp', 'a1c','chol'] # dropping highly misisng data didn't make difference
    # data.drop(columns=exclude, axis=1, inplace=True)

    train = data.sample(frac=0.8, random_state=2)
    test = data.drop(train.index)

    train_data, train_Y = dr.split_hyperparams_target(train,'stage') # split data from label
    test_data, test_Y = dr.split_hyperparams_target(test,'stage')

    final_col = train_data.columns # saving column names for later

    simp_imp = SimpleImputer(strategy='mean').fit(train_data) # impute missing values as mean
    train_imp = simp_imp.transform(train_data)
    test_imp = simp_imp.transform(test_data)

    scaler = StandardScaler().fit(train_imp) # scale values to mean 0, perserve variance
    train_clean = scaler.transform(train_imp)
    test_clean = scaler.transform(test_imp)

    train_df = pd.DataFrame(train_clean) # make things df again
    test_df = pd.DataFrame(test_clean)
    train_df.columns = final_col
    test_df.columns = final_col

    train_labels = [x[0] for x in train_Y.to_numpy()] # convert to format sklearn likes
    test_labels = [x[0] for x in test_Y.to_numpy()]

#########################################################################################

    # Now try different models
    rf = RandomForestClassifier(random_state = 0, n_estimators=100)
    knn =  KNeighborsClassifier(n_neighbors=5)
    log = LogisticRegression(random_state=0)

    models = [rf, knn,log]

    for model in models:
        model.fit(train_df, train_labels)
        predicts = model.predict(test_df)

        predicts_tensor = torch.tensor(predicts)
        test_labels_tensor = torch.tensor(test_labels)
        conf_matrix = vu.confusion_matrix(predicts_tensor, test_labels_tensor)
        classes = {0:'Not HF', 1: 'Stage C', 2: 'Stage D'}

        print(model)

        for each in classes:
            precision, recall, f1 = vu.precision_recall_F1(conf_matrix,each)
            print(f'{classes[each]}: Precision: {round(precision,2)}, Recall: {round(recall,2)}, F1: {round(f1,2)}')

        print('----------------------------------------------------------------')

def univariate():
    '''
    function to test the univariate relationships between the continuous data and the target variables
    '''
    X,y = dr.read_cont(dropna=True) # load data
    X = X.drop(columns=['id','gender','ace_arb','aldo','bb','ino','loop','arni','sglt2','stat','thia',
                                    'xanthine','albumin','bnp','smoke']) # drop non-continuous hyperparams

    train, valid, test = dr.generate_sets((X,y), splits=[70,15,15])

    #I want to drop the top/bottom 3 values for each parameter -> testing showed these ruined the plots
    train = tu.cut_topbottom(train, 100)

    tu.univariate(train, X.columns.to_list(), {0:"Stage 0", 1:"Stage 1",2:"Stage 2"})


if __name__ == '__main__':
    # univariate()
    # neural_net()
    models() # trying different models in sklearn. you guys can tweak this easily