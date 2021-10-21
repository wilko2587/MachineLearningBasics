import math


# def flatten(List):
#    '''
#    function to flatten "list", ie: turn [[1,2,3],[4,5,6]] into [1,2,3,4,5,6]
#    '''
#    # check if list is empty
#    if len(List) == 0:
#        return List
#
#    # if there is a nested list, recursively iterate
#    if isinstance(List[0], list):
#        return flatten(*List[:1]) + flatten(List[1:])
#
#    return List[:1] + flatten(List[1:])


def pctile(value, List):
    '''
    returns an approximate percentile that "value" sits at within "list" (must contain integers or floats)
    '''
    _list = list(set(List))
    _list.sort()
    N = len(_list)
    index = min(range(len(_list)), key=lambda i: abs(_list[i] - value))  # index of element closest to value in _list
    return (float(index) + 1) / float(N) * 100


def mean(data):
    '''
    returns mean of floats/integers in a list "data"
    '''
    _data = data
    _sum = sum(_data)
    return _sum / len(data)


def var(data):
    '''
    returns the variance of floats within data
    '''
    _sum2 = 0
    _data = data
    _mean = mean(_data)
    for element in _data:
        _sum2 += (element - _mean) ** 2
    return _sum2 / len(data)


def stdev(data):
    '''
    returns standard deviation of floats/integers in a list "data"
    '''
    return math.sqrt(var(data))


def greyscale_to_binary(data, threshold):
    '''
    function to turn a 2D list of floats/integers into a binary representation
    '''
    binary_data = []
    for element in data:
        if element > threshold:
            binary_data.append(1)
        else:
            binary_data.append(0)
    return binary_data


def Z_Score(data):
    '''
    Function to z-score the 2D list "data"
    '''
    Z_data = []
    _mean = mean(data)
    _std = stdev(data)
    for element in data:
        Z = (element - _mean) / _std
        Z_data.append(Z)
    return Z_data


def variance_reduce(train_data, validation_data, pctile_thresh):
    '''
    list_of_datas contains a set of training data.
    We scan through each corresponding "pixel" between the different datas, and calculate that pixel's variance across
    the dataset.
    We then eliminate pixels (uniformly for all datas in list_of_datas) that have a variance below "threshold"
    "pctile" defines the percentile of variance below which we cut
    "validation_data" is a list of other data (ie: validation/test data we want to also affect the changes on)
    NB: this works for sets of 2D lists inside list_of_dates only
    '''

    pixel_variance = []  # initialise. All values will end up being overwritten
    npixels = len(train_data[0][1])
    for x in range(
            npixels):  # scan through each pixel, and work out what the variance is for that same pixel, accross the dataset.
        pixel_data = [data[1][x] for data in train_data]
        _var = var(pixel_data)
        pixel_variance.append(_var)  # save the result in pixel_variance

    pixels_to_keep = [x for x in range(npixels) if
                      pctile(pixel_variance[x], list(set(pixel_variance))) >= pctile_thresh]

    reduced_train = [[data[0], [data[1][x] for x in pixels_to_keep]] for data in train_data]
    reduced_valid = [[data[0], [data[1][x] for x in pixels_to_keep]] for data in validation_data]

    return reduced_train, reduced_valid
