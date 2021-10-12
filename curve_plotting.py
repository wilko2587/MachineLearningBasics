import matplotlib.pyplot as plt
import ID3
import parse
import random

file = 'house_votes_84.data'
data = parse.parse(file)

noprune_curve = []
prune_curve = []
N = []
data_indices = list(range(len(data)))
for Ntraining in range(10,int(len(data)/2)):
    noprune_accs = []
    prune_accs = []
    for i in range(100): #repeat 100 times and take an average
        random.shuffle(data_indices) #shuffle the indices in place
        trainingset = [data[k] for k in data_indices[0:Ntraining]]
        testset = [data[k] for k in data_indices[Ntraining:]]

        tree = ID3.ID3(trainingset,'democrat')

        noprune_acc = ID3.test(tree,testset)

        tree = ID3.prune(tree,trainingset)
        prune_acc = ID3.test(tree,testset)

        noprune_accs.append(noprune_acc)
        prune_accs.append(prune_acc)

    ave_noprune_acc = sum(noprune_accs)/len(noprune_accs)
    ave_prune_acc = sum(prune_accs)/len(prune_accs)

    N.append(Ntraining)
    noprune_curve.append(ave_noprune_acc)
    prune_curve.append(ave_prune_acc)

plt.plot(N,noprune_curve,color='blue',label='no pruning')
plt.plot(N,prune_curve,color='red', label='pruning')
plt.xlabel('training set size')
plt.ylabel('% accuracy')
plt.legend()
plt.show()