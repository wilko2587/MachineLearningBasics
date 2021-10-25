import math

def dot(a, b):
    """
    returns dot product of vectors a,b
    """
    assert (len(a) == len(b))
    N = len(a)
    return sum([a[i] * b[i] for i in range(N)])


def mag(a):
    """
    returns magnitude of vector a
    """
    return math.sqrt(sum([x ** 2 for x in a]))


def vect_mean(List):
    """
    finds "mean" of a list of vectors List = [a1,a2,a3...], where a1 = [x1,x2,x3...] is a vector of numbers
    represented as a list.
    returns a_mean, vector with the same dimensions as a1
    """
    ndims = len(List[0])
    meanvec = []
    for i in range(ndims):
        i_mean = sum([a[i] for a in List])/len(List)
        meanvec.append(i_mean)
    return meanvec


def normalize(vect,_mins,_maxs):
    """
    function to normalise vector to a relative space between 0->1 in each dimension, based on
    that dimensions' minimum and maximum value provided in vectors _mins, _maxs
    returns normalised vector
    """
    assert (len(vect) == len(_mins) == len(_maxs))
    ndims = len(vect)

    for i in range(ndims): # deal with division by zero errors
        if _maxs[i]-_mins[i] == 0.:
            _maxs[i] += 1e-10

    return [(vect[i]-_mins[i])/(_maxs[i]-_mins[i]) for i in range(ndims)]


def softmax(x,all_xs):
    """
    returns exp(x)/ [ sum of exp(x) for x in all_xs ]
    """
    return math.exp(x)/sum([math.exp(i) for i in all_xs])


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
