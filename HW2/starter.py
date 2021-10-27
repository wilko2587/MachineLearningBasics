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
    dist = -mu.dot(a, b) / (mu.mag(a) * mu.mag(b))
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


def knn_dimred(reductionvar, train, query, metric="euclidean", k=7,
               dimred_func=tu.low_variance_filter):
    """
    wrapper for knn using dimensionally reduced data
    """
    train_lowD, query_lowD = dimred_func(train, query, reductionvar)
    return knn(k, train_lowD, query_lowD, metric=metric)


def find_centroids(train, metric_func, ncentroids=10):
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
        vec_changes = [metric_func(centroids_norm[i], old_centroids_norm[i]) for i in
                       range(ncentroids)]  # find scaled moves
        change = sum([vec_change for vec_change in
                      vec_changes]) / ncentroids  # convert to a heuristic for % overall change in all the centroids

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

    k = 10  # 10 clusters, one for each number
    # 1) iterate to find the centroids
    centroids = find_centroids(train, metric_func, ncentroids=k)

    # now label the query data
    query_results = []
    for point in query:  # now calculate the cluster for each query point
        dist_2_centroid = [metric_func(point[1], centroid[1]) for centroid in centroids]
        nearest_centroid = dist_2_centroid.index(min(dist_2_centroid))  # get index of nearest centroid (with min dist)
        query_results.append(centroids[nearest_centroid][0])  # 0th index is the label of the centroid

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

    k = 10
    # 1) iterate to find the centroids
    centroids = find_centroids(train, metric_func, ncentroids=k)

    # now label the query data
    query_results = []

    for point in query:
        dist_2_centroids = [metric_func(point[1], centroid[1]) for centroid in centroids]
        Pvalues = [mu.softmax(dist, dist_2_centroids) for dist in dist_2_centroids]
        # return these values as a dictionary, with key=centroid label, value = softmax probability
        Pkeys = [centroid[0] for centroid in centroids]
        Pdict = dict([(Pkeys[i], Pvalues[i]) for i in range(len(Pvalues))])
        query_results.append(Pdict)

    return query_results


# confusion matrix
def conf_matrix(goals, predictions):
    """
    function to return a confusion matrix for lists of "goals" and "predictions"
    """

    assert (len(goals) == len(predictions))

    conf_matrix = {}
    rows = set(goals)
    columns = set(predictions)
    for c in columns:
        conf_matrix[str(c)] = {str(r): 0 for r in rows}
    for c in columns:
        for r in rows:
            _bool = [(str(goals[i]) == str(r)) & (str(predictions[i]) == str(c)) for i in range(len(goals))]
            count = sum(_bool)
            conf_matrix[str(c)][str(r)] = count
    return conf_matrix


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


def find_KNN_optimal_params():
    train = read_data('train.csv')
    valid = read_data('valid.csv')
    train_binary = tu.data_bin(train)
    valid_binary = tu.data_bin(valid)

    print("Using our handbuilt optimization function to find the best")
    best_thresh, acc_dimred = validation_utils.voptimize(train_binary, valid_binary, knn_dimred, range(50, 90, 5),
                                                         metric="cosim",
                                                         k=7,
                                                         dimred_func=tu.low_variance_filter)

    print("best variance threshold: ", best_thresh)
    print("best accuracy: ", acc_dimred)


def run_KNN():
    train = read_data('train.csv')
    test = read_data('test.csv')
    train_binary = tu.data_bin(train)
    test_binary = tu.data_bin(test)

    print("finding accuracy of our best model: k=7, binary data and variance filter at 60%")

    # reduce the dimensionality of our data,
    train_lowD, test_lowD = tu.low_variance_filter(train_binary, test_binary, 60)
    guess_labels = knn(7, train_lowD, test_lowD, "cosim")
    true_labels = [x[0] for x in test]
    test_accuracy = validation_utils.accuracy(guess_labels, true_labels)
    print("KNN accuracy: ", test_accuracy)
    import pandas as pd
    cm = pd.DataFrame(conf_matrix(guess_labels, true_labels))
    print("KNN confusion matrix:")
    print(cm)
    cm.to_csv("KNN_cm.csv")


def run_kmeans():
    train = read_data('train.csv')
    test = read_data('test.csv')

    metric = "cosim"
    cluster_assignments = kmeans(train, test, metric)

    true_labels = [x[0] for x in test]

    import pandas as pd
    cm = pd.DataFrame(conf_matrix(true_labels, cluster_assignments))
    print("kmeans confusion matrix: ")
    print(cm)


if __name__ == "__main__":
    find_KNN_optimal_params() # Included this as we want to show our optimization function. Can take a while to run
    run_KNN()
    run_kmeans()
