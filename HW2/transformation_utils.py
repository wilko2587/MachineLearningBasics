import math_utils as mu

def data_bin(data):
    """
    function to turn data from continuous into binary
    nb: "datas" structure is
    data[0] = a string label,
    data[1] = list of numbers we will turn to binary
    returns data in the format [data[0],new_data[1]]
    """
    lin_num = 0

    for line in data: # line[0] = label, line[1] = list with all values
        pix_num = 0
        for each in line[1]: # go into each individual value
            if each > 0: # if greater than 1, replace that specific location with 1
                data [data.index(line)][1][line[1].index(each)] = 1.0

                data[lin_num][1][pix_num] = 1
            pix_num += 1
        lin_num += 1
    return data

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
    _mean = mu.mean(data)
    _std = mu.stdev(data)
    for element in data:
        Z = (element - _mean) / _std
        Z_data.append(Z)
    return Z_data


def low_variance_filter(train_data, validation_data, pctile_thresh):
    '''
    Dimensionality reduction technique.
    We scan through each corresponding "pixel" between the different datas, and calculate that pixel's variance across
    the dataset.
    We then eliminate pixels that have a variance below "threshold"
    "pctile" defines the percentile of variance below which we cut
    "validation_data" is a list of other data (ie: validation/test data we want to also affect the changes on)
    '''

    pixel_variance = []  # initialise. All values will end up being overwritten
    npixels = len(train_data[0][1])
    for x in range(
            npixels):  # scan through each pixel, and work out what the variance is for that same pixel, accross the dataset.
        pixel_data = [data[1][x] for data in train_data]
        _var = mu.var(pixel_data)
        pixel_variance.append(_var)  # save the result in pixel_variance

    pixels_to_keep = [x for x in range(npixels) if
                      mu.pctile(pixel_variance[x], list(set(pixel_variance))) >= pctile_thresh]

    reduced_train = [[data[0], [data[1][x] for x in pixels_to_keep]] for data in train_data]
    reduced_valid = [[data[0], [data[1][x] for x in pixels_to_keep]] for data in validation_data]
    print('--> {} dimensions in input; {} dimensions in output'.format(len(train_data[0][1]),len(reduced_train[0][1])))
    return reduced_train, reduced_valid


def high_correlation_filter(train_data, validation_data, correlation_threshold):
    """
    Dimensionality reduction technique
    Calculates correlations between a pixels' values vs other pixels.
    Eliminates pixels with a high correlation to any other pixels in the training data
    This function will eliminate the first pixel it comes across. Eg: if a is correlated to b, it will take the list
    [a,b,c] and return [b,c]
    """
    npixels = len(train_data[0][1])
    to_drop = [] # list to keep track of which pixels have too high correlations with the rest of the data
    for x in range(npixels):
        pixel_data = [data[1][x] for data in train_data]
        for y in range(npixels):
            if x != y and y not in to_drop:
                pearsonsR = mu.pearsons_r(pixel_data, [data[1][y] for data in train_data])
                if abs(pearsonsR) > correlation_threshold and x not in to_drop:
                    to_drop.append(x)
                    break
            else:
                pass

    reduced_train = [[data[0], [data[1][x] for x in range(npixels) if x not in to_drop]] for data in train_data]
    reduced_valid = [[data[0], [data[1][x] for x in range(npixels) if x not in to_drop]] for data in validation_data]
    print('--> {} dimensions in input; {} dimensions in output'.format(len(train_data[0][1]),len(reduced_train[0][1])))
    return reduced_train, reduced_valid
