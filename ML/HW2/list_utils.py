def format_list(list, dtype_func):
    """
    formats the elements in "list" to conform to a data type
    """
    return [dtype_func(i) for i in list]


def most_frequent(List):
    """
    returns most frequent element in a list
    """
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num
