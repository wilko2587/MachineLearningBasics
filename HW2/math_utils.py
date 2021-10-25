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
