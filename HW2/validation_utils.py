def accuracy(guess_labels, true_labels):
    """
    returns the accuracy of a set of guesses to the expected values
    """
    assert (len(guess_labels) == len(true_labels))
    N = len(guess_labels)
    return sum([int(guess_labels[i] == true_labels[i]) for i in range(N)]) / len(true_labels)  # made this a percentage


def voptimize(train, valid, func, _range,
              plot_title="",
              **kwargs):  # embedded the binary transformation in here
    """
    function to take train data "train", validation set "valid", and iterate through possible

    returns a list of tupples: (k, accuracy-of-that-k)
    NB: train and validation data is a 2d list, with each item representing an example, and each example having
    0th index a "label" and 1st index being the data.

    train = training set
    valid = validation set
    func = function to optimize, with 0th positional argument being the variable to optimize
    """
    validation_results = []
    print('Disclaimer: voptimize can take a while to run using our standard libraries... hold tight')
    for x in _range:
        valid_guess = func(x, train, valid, **kwargs)  # run the function
        validation_true = [i[0] for i in valid]
        valid_accuracy = accuracy(valid_guess, validation_true)
        validation_results.append(valid_accuracy)
        print(f'x: {x}, accuracy: {valid_accuracy}')

    try:  # if matplotlib available, plot the training and validation accuracies on a figure. If not, pass
        import matplotlib.pyplot as plt
        plt.plot(_range, validation_results)
        plt.title(plot_title)
    except:
        print("looks like matplotlib problems... No chart for you...")

    best_accuracy = max(validation_results)
    best_var = _range[validation_results.index(best_accuracy)]
    return best_var, best_accuracy
