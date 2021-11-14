import pandas as pd


def confusion_matrix(expected, predicted, classes=[0,1,2]):
    '''
    builds a confucion matrix for expected and predicted values using pandas dataframe

    :expected: 1D tensor of expected values
    :predicted: 1D tensor of predicted values
    :returns: pandas dataframe
    '''

    assert (len(predicted) == len(expected))

    expected = expected.tolist()
    predicted = predicted.tolist()

    # initialise the confusion matrix
    cfm = pd.DataFrame(data=0,
                       columns=classes,
                       index=classes)

    for i in range(len(predicted)):  # build the values in the confusion matrix
        e = expected[i]
        p = predicted[i]
        cfm.loc[p, e] += 1
    return cfm


def precision_recall_F1(cfm, _class):
    '''
    quick function to return the precision of a class from a confusion matrix

    :param: cfm = pandas dataframe (the confusion matrix)
    :param: _class = the class that we're measuring with respect to
        see https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
    :returns: precision, recall, F1 (decimal 0->1)
    '''

    tp = cfm.loc[_class,_class]
    tn = cfm.drop(columns=_class,index=_class).sum()
    fp = cfm.drop(columns=_class).loc[_class].sum()
    fn = cfm.drop(index=_class).loc[:,_class].sum()

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)

    return precision, recall, f1
