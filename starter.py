import math


# returns Euclidean distance between vectors a dn b
def euclidean(a, b):
    c = []
    for i in range(0, len(a) - 1):
        c.append((a[i] - b[i]) ** 2)
    sum1 = sum(c)
    dist = math.sqrt(sum1)
    return (dist)


euclidean([1, 1, 2], [2, 1, 3])

# returns Cosine Similarity between vectors a dn b
import math


def cosim(a, b):
    dot1 = []
    mag_a = []
    mag_b = []
    for i in range(0, len(a) - 1):
        dot1.append(a[i] * b[i])
        mag_a.append((a[i]) ** 2)
        mag_b.append((b[i]) ** 2)
    sum_dot1 = sum(dot1)
    sum_mag_a = sum(mag_a)
    sum_mag_b = sum(mag_b)
    root_a = math.sqrt(sum_mag_a)
    root_b = math.sqrt(sum_mag_b)
    dist = sum_dot1 / (root_a * root_b)
    return (dist)


cosim([1, 1, 2], [2, 1, 3])
# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    return(labels)

# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    return(labels)

def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    show('valid.csv','pixels')
    
if __name__ == "__main__":
    main()
    