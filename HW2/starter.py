import math

#returns dot product of vectors a,b
def dot(a,b):
    assert(len(a)==len(b))
    N = len(a)
    return sum([a[i]*b[i] for i in range(N)])

#returns magnitude of vector a
def mag(a):
    return math.sqrt(sum([x**2 for x in a]))

# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    assert(len(a)==len(b))
    N = len(a)
    dist2 = sum([(a[i]-b[i])**2 for i in range(N)])
    dist = math.sqrt(dist2)
    return(dist)
        
# returns Cosine Similarity between vectors a dn b
def cosin(a,b):
    dist = dot(a,b)/(mag(a)*mag(b))
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
    