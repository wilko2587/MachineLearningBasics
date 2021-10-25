'''
script to use training set and validation set to find the optimal values for
1)  K in KNN
2)  The variance threshold for discarding pixels ("pctile_thresh" in variance_reduce() )
'''

from starter import read_data, data_bin, knn,
from starter import data_bin

def best_k(): #embedded the binary transformation in here
    '''
    Starts with k = 1 then will increase k and keep printing accuracy testing on validation data until arbitrary stop.
    I did until k > 50 just to see.
    Also did until accuracy falls but seems highest accuracy is 0.86 (k = 1).
    Can someone make this print out more sig figs??
    For whatever reason on validation set, accuracy is super similar until k = 9 then starts to fall.
    I ran this on test set to compare with James's output and its the same.
    '''
    k = 1
    k_output = list() # append tuple with (k value, validation accuracy)

    train = read_data('train.csv')
    train = data_bin(train)
    valid = read_data('valid.csv')
    valid = data_bin(valid)

    while True:
        labels = knn(train, valid, "euclidean",k)
        true_labels = [x[0] for x in valid]
        _accuracy = accuracy(labels,true_labels)
        print(f'k: {k}, accuracy: {_accuracy}')
        k_output.append((k,_accuracy))

        if k == 1:
            pass
        elif k == 10: # or _accuracy < k_output[-2][1]
            break

        k = k + 1

    return k_output