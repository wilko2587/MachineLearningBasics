import matplotlib.pyplot as plt

def plot_validation_accuracy(xylist,title = None):

    x = [i[0] for i in xylist]
    y = [i[1] for i in xylist]

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(x,y,label = 'validation accuracy')
    plt.legend()
    plt.title(title)
    plt.show()
    return

