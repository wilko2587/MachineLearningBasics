import math
import math_utils as mu
import list_utils as lu
import transformation_utils as tu
import validation_utils


def euclidean(a, b):
    """
    returns Euclidean distance between vectors a dn b
    """
    assert (len(a) == len(b))
    N = len(a)
    dist2 = sum([(a[i] - b[i]) ** 2 for i in range(N)])
    dist = math.sqrt(dist2)
    return (dist)


def cosim(a, b):
    """
    returns Cosine Similarity between vectors a dn b
    """
    dist = mu.dot(a, b) / (mu.mag(a) * mu.mag(b))
    return (dist)


def knn(k, train, query, metric):
    """
    returns a list of labels for the query dataset based upon labeled observations in the train dataset.
    metric is a string specifying either "euclidean" or "cosim".
    All hyper-parameters should be hard-coded in the algorithm.
    """

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

        label = lu.most_frequent(all_labels)
        labels.append(label)

    return (labels)


def knn_dimred(pctthresh, train, valid, metric="euclidean", k=7):
    """
    little wrapper for knn using dimensionally reduced data
    """
    train_lowD, valid_lowD = tu.variance_reduce(train, valid, pctile_thresh=pctthresh)
    return knn(k, train_lowD, valid_lowD, metric=metric)


def kmeans(train, query, metric):
    """
    returns a list of labels for the query dataset based upon observations in the train dataset.
    labels should be ignored in the training set
    metric is a string specifying either "euclidean" or "cosim".
    All hyper-parameters should be hard-coded in the algorithm.
    """
    return


def read_data(file_name):
    """
    function to read data from file_name
    :returns data in a tupple
    """
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = float(tokens[0])
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            data_set.append([label, lu.format_list(attribs, float)])
    return (data_set)


def show(file_name, mode):
    """
    function to print the image held in data from file_name.
    Prints a blank space in terminal if a pixel is "0", else "*"
    """
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


def main():
    # show('test.csv', 'pixels')
    train = read_data('train.csv')
    valid = read_data('valid.csv')
    test = read_data('test.csv')

    validation_utils.voptimize(train, valid, knn, range(1, 20, 1),
                               plot_title="KNN accuracy by different k",
                               metric="euclidean")

    train_binary = tu.data_bin(train)
    valid_binary = tu.data_bin(valid)

    validation_utils.voptimize(train_binary, valid_binary, knn, range(1, 20, 1),
                               plot_title="Binary KNN accuracy by different k",
                               metric="euclidean")

    validation_utils.voptimize(train_binary, valid_binary, knn_dimred, range(30,70,5),
                               plot_title="Binary, Dim-reduced KNN accuracy by dimensionality reduction level (0-100%)",
                               metric="euclidean")

    # labels = knn(train, test, "euclidean")
    # true_labels = [x[0] for x in test]
    # _accuracy = accuracy(labels,true_labels)
    # print(f'accuracy: {_accuracy}')


if __name__ == "__main__":
    main()

# Left to do:
# 10x10 confusion matrix
# k means classifier
# soft k means classifier
# collaborative filter question
# write-up
