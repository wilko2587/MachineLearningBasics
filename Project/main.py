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
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingClassifier
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

def models():
    '''
    I removed id (not needed for any analysis), as well as bnp, a1c, and chol because of high degree of missingness.
    Transformed gender and smoking to numeric.
    Imputed missing values as mean. Scaled data to mean 0, same SD as prior.
    Ran 5 models: baseline RF, RF with not important elements removed, KNN, Gaussian mixture model, and gradient boosting classifier
    All stink.
    '''

    # Load data, drop irrelevant or missing, convert to numeric
    data = dr.read('deid_full_data_cont.csv')
    exclude = ['id','bnp','a1c','chol'] # drop id because it isn't useful, values with high missingness
    data.drop(columns=exclude, axis=1, inplace=True)
    data['gender'].replace(to_replace=['Male', 'Female',], value=[0, 1], inplace=True) # make numeric
    smoke_cat = data['smoke'].unique()
    data['smoke'].replace(to_replace=smoke_cat, value=np.arange(11), inplace=True)

    # 80/20 train/test split
    train = data.sample(frac=0.8, random_state=2)
    test = data.drop(train.index)

    train_data, train_Y = dr.split_hyperparams_target(train,'stage')
    test_data, test_Y = dr.split_hyperparams_target(test,'stage')

    final_col = train_data.columns # saving column names for later

    # Putting labels in a format sklearn needs
    train_labels = [x[0] for x in train_Y.to_numpy()]
    test_labels = [x[0] for x in test_Y.to_numpy()]

    # Imputing values
    simp_imp = SimpleImputer(strategy='mean').fit(train_data)
    train_imp = simp_imp.transform(train_data)
    test_imp = simp_imp.transform(test_data)

    # Scaling
    scaler = StandardScaler().fit(train_imp)
    train_clean = scaler.transform(train_imp)
    test_clean = scaler.transform(test_imp)

    # Putting things back as a df
    train_df = pd.DataFrame(train_clean)
    test_df = pd.DataFrame(test_clean)
    train_df.columns = final_col
    test_df.columns = final_col

    # RF model
    rf = RandomForestClassifier(random_state = 0, n_estimators=100)
    rf.fit(train_df,train_labels)
    predicts = rf.predict(test_df)

    # Model Performance
    predicts_tensor = torch.tensor(predicts)
    test_labels_tensor = torch.tensor(test_labels)
    conf_matrix = vu.confusion_matrix(predicts_tensor, test_labels_tensor)
    classes = {0:'Not HF', 1: 'Stage C', 2: 'Stage D'}

    print('Baseline RF')
    for each in classes:
        precision, recall, f1 = vu.precision_recall_F1(conf_matrix,each)
        print(f'{classes[each]}: Precision: {round(precision,2)}, Recall: {round(recall,2)}, F1: {round(f1,2)}')

    # RF model 2 - tried removing unimportant features, didn't help
    not_important = list()

    for x,y in zip(train_df,rf.feature_importances_):
        if y < 0.05:
            not_important.append(x)

    train_trim_df = train_df.drop(not_important,axis=1)
    test_trim_df = test_df.drop(not_important,axis=1)
    rf2 = RandomForestClassifier(random_state = 0, n_estimators=100)
    rf2.fit(train_trim_df,train_labels)
    predicts2 = rf2.predict(test_trim_df)

    # Trimmed RF Performance
    predicts_tensor2 = torch.tensor(predicts2)
    conf_matrix2 = vu.confusion_matrix(predicts_tensor2, test_labels_tensor)

    print('RF trimmed')

    for each in classes:
        precision, recall, f1 = vu.precision_recall_F1(conf_matrix2,each)
        print(f'{classes[each]}: Precision: {round(precision,2)}, Recall: {round(recall,2)}, F1: {round(f1,2)}')

    # K nearest neighbors (k=5, all choices were bad)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_df,train_labels)
    predicts3 = neigh.predict(test_df)

    # KNN Performance
    predicts_tensor3 = torch.tensor(predicts3)
    test_labels_tensor3 = torch.tensor(test_labels)
    conf_matrix3 = vu.confusion_matrix(predicts_tensor3, test_labels_tensor3)

    print('k=5')

    for each in classes:
        precision, recall, f1 = vu.precision_recall_F1(conf_matrix3, each)
        print(f'{classes[each]}: Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, F1: {round(f1, 2)}')

    # Gaussian mixture model
    gaus = GaussianMixture(n_components=3,random_state=0)
    gaus.fit(train_df,train_labels)
    predicts4 = gaus.predict(test_df)

    # Model Performance
    predicts_tensor4 = torch.tensor(predicts4)
    conf_matrix4 = vu.confusion_matrix(predicts_tensor4, test_labels_tensor)

    print('Gaussian mixture model')

    for each in classes:
        precision, recall, f1 = vu.precision_recall_F1(conf_matrix4, each)
        print(f'{classes[each]}: Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, F1: {round(f1, 2)}')

    # Gradient boosting classifier
    gbm = GradientBoostingClassifier(n_estimators=1000, learning_rate=1, max_depth=3, random_state=0)
    gbm.fit(train_df,train_labels)
    predicts5 = gbm.predict(test_df)

    # Model Performance
    predicts_tensor5 = torch.tensor(predicts5)
    conf_matrix5 = vu.confusion_matrix(predicts_tensor5, test_labels_tensor)

    print('gradient boosting classifier')

    for each in classes:
        precision, recall, f1 = vu.precision_recall_F1(conf_matrix4, each)
        print(f'{classes[each]}: Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, F1: {round(f1, 2)}')

if __name__ == '__main__':
    # neural_net()
    models() #RF, RF trimmed, KNN, Gaussian mixture model, gradient boosting classifier