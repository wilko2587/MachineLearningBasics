import math
import math_utils as mu
import list_utils as lu
import transformation_utils as tu
import validation_utils
import random


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


def knn_dimred(pctthresh, train, query, metric="euclidean", k=7):
    """
    wrapper for knn using dimensionally reduced data
    """
    train_lowD, query_lowD = tu.variance_reduce(train, query, pctile_thresh=pctthresh)
    return knn(k, train_lowD, query_lowD, metric=metric)


def find_centroids(train, metric_func, ncentroids = 7):
    """
    function to iteratively find centroids for the kmeans algorithm
    initialises a number of "ncentroids" centroids
    returns the coordinates of each of those centroids
    """
    ndims = len(train[0][1])
    _mins = [min([x[1][i] for x in train]) for i in range(ndims)]  # minimum value for each pixel
    _maxs = [max([x[1][i] for x in train]) for i in range(ndims)]  # maximum value for each pixel

    # initialize the centroids on top of randomly chosen points in the dataset

    centroid_labels = ['a', 'b', 'c', 'd', 'e',
                      'f', 'g', 'h', 'i', 'j',
                      'k', 'l', 'm', 'n', 'o',
                      'p', 'q', 'r', 's', 't',
                      'u', 'v', 'w', 'x', 'y', 'z']

    centroids = []
    for i in range(ncentroids):
        init = train[random.randint(0, ndims)][1]
        centroids.append([centroid_labels[i], init])



    change = 100
    counter = 0
    while change > 0.05 and counter < 50:  # first, iterate the centroids until convergence or max limit reached
        old_centroids = centroids.copy()
        for point in train:
            dist_2_centroids = [metric_func(point[1], centroid[1]) for centroid in centroids]
            nearest_centroid = dist_2_centroids.index(
                min(dist_2_centroids))  # get index of nearest centroid (with min dist)
            point[0] = centroids[nearest_centroid][0]  # update the label of the datapoint to match its closest centroid

        for i in range(ncentroids):
            nearest_points = [point[1] for point in train if point[0] == centroids[i][0]]
            new_centroid = mu.vect_mean(nearest_points)
            centroids[i][1] = new_centroid

        # scale the centroid vectors so all dimensions fall between 0->1, use this to judge change
        centroids_norm = [mu.normalize(centroid[1], _mins, _maxs) for centroid in centroids]
        old_centroids_norm = [mu.normalize(centroid[1], _mins, _maxs) for centroid in old_centroids]
        vec_changes = [metric_func(centroids_norm[i], old_centroids_norm[i]) for i in range(ncentroids)]  # find scaled moves
        change = sum([vec_change for vec_change in vec_changes]) / ncentroids  # convert to a % change overall
        counter += 1

    return centroids


def kmeans(train, query, metric):
    """
    returns a list of labels for the query dataset based upon observations in the train dataset.
    labels should be ignored in the training set
    metric is a string specifying either "euclidean" or "cosim".
    All hyper-parameters should be hard-coded in the algorithm.
    """

    if metric == "euclidean":
        metric_func = euclidean
    elif metric == "cosim":
        metric_func = cosim
    else:
        raise NameError("Invalid Metric choice")

    k=7
    #1) iterate to find the centroids
    centroids = find_centroids(train,metric_func,ncentroids=k)

    # now label the query data
    query_results = []
    for point in query:  # now calculate the cluster for each query point
        dist_2_centroid = [metric_func(point[1], centroid[1]) for centroid in centroids]
        nearest_centroid = dist_2_centroid.index(min(dist_2_centroid))  # get index of nearest centroid (with min dist)
        query_results.append(centroids[nearest_centroid][0]) #0th index is the label of the centroid

    return query_results


def soft_kmeans(train, query, metric):
    """
    function to perform soft k-means. Uses our existing kmeans() function
    generates probabilities of labels for each query using softmax
    """
    if metric == "euclidean":
        metric_func = euclidean
    elif metric == "cosim":
        metric_func = cosim
    else:
        raise NameError("Invalid Metric choice")

    k=7
    #1) iterate to find the centroids
    centroids = find_centroids(train,metric_func,ncentroids=k)

    # now label the query data
    query_results = []

    for point in query:
        dist_2_centroids = [metric_func(point[1], centroid[1]) for centroid in centroids]
        Pvalues = [mu.softmax(dist,dist_2_centroids) for dist in dist_2_centroids]
        #return these values as a dictionary, with key=centroid label, value = softmax probability
        Pkeys = [centroid[0] for centroid in centroids]
        Pdict = dict([(Pkeys[i],Pvalues[i]) for i in range(len(Pvalues))])
        query_results.append(Pdict)

    return query_results


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

    # 1) euclidean KNN
    print('\n running KNN on data with euclidean metric...')
    bestk, acc = validation_utils.voptimize(train, valid, knn, range(1, 20, 1),
                                            plot_title="KNN accuracy by different k",
                                            metric="euclidean")

    testset_guess = knn(bestk, train, test, metric="euclidean")
    true_labels = [x[0] for x in test]
    test_acc = validation_utils.accuracy(testset_guess, true_labels)

    print("best euclidean KNN accuracy with k = {} \n".format(bestk),
          "validation accuracy = {} \n".format(acc),
          "test accuracy = {}".format(test_acc))

    # 2) cosine KNN
    print('\n running KNN on data with cosim metric...')
    bestk, acc = validation_utils.voptimize(train, valid, knn, range(1, 20, 1),
                                            plot_title="KNN accuracy by different k",
                                            metric="cosim")

    testset_guess = knn(bestk, train, test, metric="cosim")
    test_labels = [x[0] for x in test]
    test_acc = validation_utils.accuracy(testset_guess, test_labels)

    print("best cosim KNN accuracy with k = {} \n".format(bestk),
          "validation accuracy = {} \n".format(acc),
          "test accuracy = {}".format(test_acc))

    # 3) Greyscale -> Binary conversion
    print('\n running KNN on data with euclidean metric, converting greyscale -> binary...')
    train_binary = tu.data_bin(train)
    valid_binary = tu.data_bin(valid)
    test_binary = tu.data_bin(test)

    bestk_binary, acc_binary = validation_utils.voptimize(train_binary, valid_binary, knn, range(1, 20, 1),
                                                          plot_title="Binary KNN accuracy by different k",
                                                          metric="euclidean")

    testset_guess_binary = knn(bestk_binary, train_binary, test_binary, metric="euclidean")
    test_acc_binary = validation_utils.accuracy(testset_guess_binary, test_labels)

    print("best binary KNN accuracy with k = {} \n".format(bestk_binary),
          "validation accuracy = {} \n".format(acc_binary),
          "test accuracy = {}".format(test_acc_binary))

    # 4) Introduce dimensionality reduction, removing low variance pixels below a certain pctile
    print('\n running KNN on data with euclidean metric, converting greyscale -> binary, and eliminating low-variance pixels...')
    best_thresh, acc_dimred = validation_utils.voptimize(train_binary, valid_binary, knn_dimred, range(30, 70, 5),
                                                         plot_title="Binary, Dim-reduced KNN accuracy by "
                                                                    "dimensionality reduction level (0-100%)",
                                                         metric="euclidean",
                                                         k=bestk_binary)

    testset_guess_dimred = knn_dimred(best_thresh, train_binary, test_binary, metric="euclidean")
    test_acc_dimred = validation_utils.accuracy(testset_guess_dimred, test_labels)

    print("best binary KNN accuracy by removing pixels with variance lower than {} pctile \n".format(best_thresh),
          "validation accuracy = {} \n".format(acc_dimred),
          "test accuracy = {}".format(test_acc_dimred))

    #guess = kmeans(train, test, "euclidean")


if __name__ == "__main__":
    main()

# Left to do:
# 10x10 confusion matrix
# k means classifier
# soft k means classifier
# collaborative filter question
# write-up
