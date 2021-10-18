import numpy as np
from numpy.linalg import norm
import math

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    l = []
    sums = 0
    for i in range(len(a)):
#       print('i', i)
        diff = a[i]-b[i]
#       print('Diff',diff)
        square = math.pow(diff,2)
#       print('Sq',square)
        sums += square
#       print('Adding to sum',sums)
        dist = math.sqrt(sums)
    #   print(dist)
    return dist

def euclidean_np(a,b):
    a = np.array(a)
    b = np.array(b)
    diff = a-b
    print(diff)
    square = np.square(diff)
    print(square)
    sums = sum(square)
    print(sums)
    dist = np.sqrt(np.sum(np.square(a-b)))
    print(dist)
    return(dist)
        
# returns Cosine Similarity between vectors a dn b
def cosim(a, b):
    num_sum, sum_a, sum_b = 0, 0, 0

    for i in range(len(a)):
        # print('i:', i)
        num = a[i] * b[i]
        # print('a[i]*b[i]:', num)
        num_sum += num
        # print('num_sum:', num_sum)
        a_sq = math.pow(a[i], 2)
        b_sq = math.pow(b[i], 2)
        # print('a_sq:',a_sq,'\n','b_sq:',b_sq)
        sum_a += a_sq
        # print('sum_a:', sum_a)
        root_a = math.sqrt(sum_a)
        # print('root_a:', root_a)
        sum_b += b_sq
        # print('sum_b:', sum_b)
        root_b = math.sqrt(sum_b)
        # print('root_b:', root_b)
        den = root_a * root_b
        # print('Denominator:', den)
        dist = num_sum / den
        # print('dist:',dist)
    return (dist)

def cosim_np(a,b):
    norm_a = norm(a)
    print('Norm_a:', norm_a)
    norm_b = norm(b)
    print('Norm_b:', norm_b)
    den = norm(a)*norm(b)
    print('Den:', den)
    dot_pdt = np.dot(a,b)
    print('Dot(a,b):', np.dot(a,b))
    dist = dot_pdt/(norm(a)*norm(b))
    return(dist)

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
    