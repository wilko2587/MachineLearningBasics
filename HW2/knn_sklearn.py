import math
from sklearn.neighbors import KNeighborsClassifier
from starter import accuracy, read_data

'''
For fun I made a KNN classifier using scikit learn. 
It's like 500x faster and easier implement, but we get similar accuracies!
'''

if __name__ == '__main__':

    train = read_data('train.csv')
    test = read_data('test.csv')

    train_data = list()
    train_labels = list()
    test_data = list()
    test_labels = list()

    for each in train:
        train_data.append(each[1])

    for each in train:
        train_labels.append(each[0])

    for each in test:
        test_data.append(each[1])

    for each in test:
        test_labels.append(each[0])

    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(train_data, train_labels)

    predict_label = neigh.predict(test_data)

    _accuracy = accuracy(predict_label, test_labels)

    print(f'accuracy: {_accuracy}')


