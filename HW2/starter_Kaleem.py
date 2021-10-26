import numpy as np
from numpy.linalg import norm
import math
#
#
#
# # returns Euclidean distance between vectors a dn b
# def euclidean(a, b):
#     l = []
#     sums = 0
#     for i in range(len(a)):
#         #       print('i', i)
#         diff = a[i] - b[i]
#         #       print('Diff',diff)
#         square = math.pow(diff, 2)
#         #       print('Sq',square)
#         sums += square
#         #       print('Adding to sum',sums)
#         dist = math.sqrt(sums)
#     #   print(dist)
#     return dist
#
# # alternatively take two np arrays, subtract and square if not using np.linearalg.norm
#
# # returns Cosine Similarity between vectors a dn b
# def cosim(a, b):
#     num_sum, sum_a, sum_b = 0, 0, 0
#
#     for i in range(len(a)):
#         # print('i:', i)
#         num = a[i] * b[i]
#         # print('a[i]*b[i]:', num)
#         num_sum += num
#         # print('num_sum:', num_sum)
#         a_sq = math.pow(a[i], 2)
#         b_sq = math.pow(b[i], 2)
#         # print('a_sq:',a_sq,'\n','b_sq:',b_sq)
#         sum_a += a_sq
#         # print('sum_a:', sum_a)
#         root_a = math.sqrt(sum_a)
#         # print('root_a:', root_a)
#         sum_b += b_sq
#         # print('sum_b:', sum_b)
#         root_b = math.sqrt(sum_b)
#         # print('root_b:', root_b)
#         den = root_a * root_b
#         # print('Denominator:', den)
#         dist = num_sum / den
#         # print('dist:',dist)
#     return (dist)
#
# #
# # # returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# # # metric is a string specifying either "euclidean" or "cosim".
# # # All hyper-parameters should be hard-coded in the algorithm.
# # def knn(train, query, metric):
# #     distance = list()
# #     for i in train:
# #         dist = euclidean(query, i)
# #         distance.append((i, dist))
# #     distance.sort(key = lambda tup: tup[1])
# #     labels = list()
# #     for j in range(metric):
# #         labels.append(distance[j][0])
# #     return (labels)
# #
# #
# # def knn_np(train, query, metric):
# #     distance = list()
# #     data = []
# #     for i in train:
# #         dist = euclidean(query, i)
# #         distance.append(i)
# #     distance = np.array(distance)
# #     data = np.array(data)
# #     index_dist = distance.argsort()
# #     data = data[index_dist]
# #     labels = data[:metric]
# #
# #     return (labels)
# #
#
# # returns a list of labels for the query dataset based upon observations in the train dataset.
# # labels should be ignored in the training set
# # metric is a string specifying either "euclidean" or "cosim".
# # All hyper-parameters should be hard-coded in the algorithm.
#
# # def kmeans(train, query, metric):
# #     return (labels)
#
# # setup class for K Means
# class Kmeans:
#     # set k, tolerance and max iterations to undergo
#     def __init__(self, n_classes, tolerance = 0.0001, max_iterations = 1000):
#         self.n_classes = n_classes
#         self.tolerance = tolerance
#         self.max_iterations = max_iterations
#
#     def fit(self, features):
#         # create centroids
#         self.centroids = {}
#         # initialize centroids (consider using import random)
#         for i in range(self.n_classes):
#             self.centroids[i] = features[i]
#
#         # initialize loop iterations
#         for i in range(self.max_iterations):
#             self.classification = {}
#
#             for i in range(self.k):
#                 self.classification[i] = []
#
#             # choose nearest centroid by finding the distance between the point and cluster
#             for points in features:
#                 # numpy version
#                 # distances = [np.linalg.norm(points - self.centroids[centroid]) for centroid in self.centroids]
#                 # classification = distances.index(min(distances))
#                 # self.classes[classification].append(points)
#
#                 # my attempt to change that lol
#                 distance = []
#                 if distance == "euclidean":
#                    metric_func = euclidean
#                 elif distance == "cosim":
#                    metric_func = cosim
#                 else:
#                     raise NameError("Invalid Metric choice")
#
#                 for labelledpoint in centroids:
#                     distance.append(metric_func(points[1], labelledpoint[1]))
#                     classification = distance.index(min(distance))
#                     self.classes[classification].append(points)
#
#                 previous = {(self.centroids)}
#
#             # recalculate the centroids by averaging the data points
#             for classification in self.classes:
#                 # numpy version
#                 # self.centroids[classification] = np.average(self.classes[classification], axis=0)
#
#                 # my attempt lol
#                 self.centroids[classification] = sum(classification) / len(classification)
#
#             isOptimal = True
#
#             for centroid in self.centroids:
#                 original_centroid = previous[centroid]
#                 current = self.centroids[centroid]
#
#                 if sum((current - original_centroid)/original_centroid * 100.0) > self.tolerance:
#                     print(sum((current - original_centroid)/original_centroid * 100.0))
#                     isOptimal = False
#
#             if isOptimal:
#                 break
#
#     def predict(self, features):
#         # numpy version
#         # distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
#         # classification = distances.index(min(distances))
#         # return classification
#
#         # my attempt lol
#         distance = []
#
#         if distance == "euclidean":
#             metric_func = euclidean
#         elif distance == "cosim":
#             metric_func = cosim
#         else:
#             raise NameError("Invalid Metric choice")
#
#         distance.append(metric_func(points[1], self.centroids[1]))
#         classification = distance.index(min(distance))
#         return classification
#
#
#
# def read_data(file_name):
#     data_set = []
#     with open(file_name, 'rt') as f:
#         for line in f:
#             line = line.replace('\n', '')
#             tokens = line.split(',')
#             label = tokens[0]
#             attribs = []
#             for i in range(784):
#                 attribs.append(tokens[i + 1])
#             data_set.append([label, attribs])
#     return (data_set)
#
#
# def show(file_name, mode):
#     data_set = read_data(file_name)
#     for obs in range(len(data_set)):
#         for idx in range(784):
#             if mode == 'pixels':
#                 if data_set[obs][1][idx] == '0':
#                     print(' ', end='')
#                 else:
#                     print('*', end='')
#             else:
#                 print('%4s ' % data_set[obs][1][idx], end='')
#             if (idx % 28) == 27:
#                 print(' ')
#         print('LABEL: %s' % data_set[obs][0], end='')
#         print(' ')
#
#
# def main():
#     show('valid.csv', 'pixels')
#
#
# if __name__ == "__main__":
#     main()

# confusion matrix
def conf_matrix(goals, predictions):
    conf_matrix = {}
    group = set(goals)
    for j in group:
        conf_matrix[j] = {i:0 for i in group}
        print(j)
    print(conf_matrix)
    for k in range(len(goals)):
        conf_matrix[goals[k]][predictions[k]] += 1
    return conf_matrix

dummy_goals = ['a', 'b', 'c', 'd', 'a', 'a', 'd']
dummy_predictions = ['a', 'b', 'b', 'a', 'a', 'a', 'd']


print(conf_matrix(dummy_goals, dummy_predictions))