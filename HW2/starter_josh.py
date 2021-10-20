# This is built on James's code
# Made accuracy a percentage

import math


# formats the elements in "list" to conform to a data type
def format_list(list, dtype_func):
    return [dtype_func(i) for i in list]


# returns dot product of vectors a,b
def dot(a, b):
    assert (len(a) == len(b))
    N = len(a)
    return sum([a[i] * b[i] for i in range(N)])


# returns most frequent element in a list
def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num


# returns magnitude of vector a
def mag(a):
    return math.sqrt(sum([x ** 2 for x in a]))


# returns Euclidean distance between vectors a dn b
def euclidean(a, b):
    assert (len(a) == len(b))
    N = len(a)
    dist2 = sum([(a[i] - b[i]) ** 2 for i in range(N)])
    dist = math.sqrt(dist2)
    return (dist)


# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    dist = dot(a, b) / (mag(a) * mag(b))
    return (dist)


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train, query, metric):
    k = 4  # number of nearest neighbors to look at

    if metric == "euclidean":
        metric_func = euclidean
    elif metric == "cosim":
        metric_func = cosim
    else:
        raise NameError("Invalid Metric choice")

    labels = []
    for testpoint in query:
        traincopy = train.copy()
        dists = []

        for labelledpoint in train:
            dists.append(metric_func(testpoint[1], labelledpoint[1]))

        all_labels = []
        while len(all_labels) < k:
            next_nearest_index = dists.index(min(dists))  # index of the nearest training point
            all_labels.append(traincopy[next_nearest_index][0])
            # now remove the smallest value from dists and train
            traincopy.pop(next_nearest_index)
            dists.pop(next_nearest_index)

        label = most_frequent(all_labels)
        labels.append(label)

    return (labels)


# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.

def kmeans(train, query, metric):
    return (labels)


def read_data(file_name):
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = float(tokens[0])
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            data_set.append([label, format_list(attribs, float)])
    return (data_set)


def show(file_name, mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % data_set[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0], end='')
        print(' ')


# returns the accuracy of a set of guesses to the expected values
def accuracy(guess_labels,true_labels):
    assert(len(guess_labels) == len(true_labels))
    N = len(guess_labels)
    return sum([int(guess_labels[i]==true_labels[i]) for i in range(N)]) / len(true_labels) # made this a percentage

'''
Starts with k = 1 then will increase k and keep printing accuracy testing on validation data until arbitrary stop.
I did until k > 50 just to see.
Also did until accuracy falls but seems highest accuracy is 0.86 (k = 1).
Can someone make this print out more sig figs??
For whatever reason on validation set, accuracy is super similar until k = 9 then starts to fall. 
I ran this on test set to compare with James's output and its the same. 
'''
def best_k():
    k = 1
    k_output = list() # append tuple with (k value, validation accuracy)

    train = read_data('train.csv')
    valid = read_data('valid.csv')
    test = read_data('test.csv')

    while True:
        labels = knn_bestk(train, valid, "euclidean",k)
        true_labels = [x[0] for x in valid]
        _accuracy = accuracy(labels,true_labels)
        print(f'k: {k}, accuracy: {_accuracy}')
        k_output.append((k,_accuracy))

        if k == 1:
            pass
        elif k > 10: # or _accuracy < k_output[-2][1]
            break

        k = k + 1

    return k_output


# Same function as James's but takes k as input instead of hard coding
# Used this for my best_k function
def knn_bestk(train, query, metric,k):
    k_val = k

    if metric == "euclidean":
        metric_func = euclidean
    elif metric == "cosim":
        metric_func = cosim
    else:
        raise NameError("Invalid Metric choice")

    labels = []
    for testpoint in query:
        traincopy = train.copy()
        dists = []

        for labelledpoint in train:
            dists.append(metric_func(testpoint[1], labelledpoint[1]))

        all_labels = []
        while len(all_labels) < k_val:
            next_nearest_index = dists.index(min(dists))  # index of the nearest training point
            all_labels.append(traincopy[next_nearest_index][0])
            # now remove the smallest value from dists and train
            traincopy.pop(next_nearest_index)
            dists.pop(next_nearest_index)

        label = most_frequent(all_labels)
        labels.append(label)

    return (labels)

def main():
    # show('test.csv', 'pixels')
    train = read_data('train.csv')
    valid = read_data('valid.csv')
    test = read_data('test.csv')
    labels = knn(train, test, "euclidean")
    true_labels = [x[0] for x in test]
    _accuracy = accuracy(labels,true_labels)
    print(f'accuracy: {_accuracy}') # made this an f string and added %


if __name__ == "__main__":
    # main()
    best_k()

# decide on transformations to speed this up
# build k means clustering

# validation set vs K, keep increasing K until validation set accuracy falls