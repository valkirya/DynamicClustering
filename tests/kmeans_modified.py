
"""
K-means variation that produces clusters of the same size.

The basic idea of the algorithm is as follows:
Compute the desired cluster size
Initialize means, preferably with k-means++
Order points by the distance to their nearest cluster minus distance 
to the farthest cluster (= biggest benefit of best over worst assignment)
Assign points to their preferred cluster until this cluster is full, then 
resort remaining objects, without taking the full cluster into account anymore
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from itertools import cycle, islice
import matplotlib.pyplot as plt

data = np.array(range(20)).reshape(2,10).T
data[2][0] = 1
data[9][0] = 100

mat = data
clsize = 2
k = 5
nrow = data.shape[0]

kmeans = MiniBatchKMeans(n_clusters = k,
max_iter = 100000,
batch_size = 750,
 max_no_improvement = 10)

kmeans.fit(mat) 

###
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
y_pred = kmeans.labels_.astype(np.int)
plt.title('kmeans', size=18)
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))

colors = np.append(colors, ["#000000"])
plt.scatter(data[:, 0], data[:, 1], s=10, color=colors[y_pred])

####

labs = np.array([None]*nrow)
clsizes = np.array([0]*k)

centroids_dist = pairwise_distances(data, kmeans.cluster_centers_)
method = 'mind'

if method == 'bova': #biggest benefit of best over worst assignment
    order_index = np.argsort(np.min(centroids_dist,axis=1) - np.max(centroids_dist,axis=1)) 
elif method == 'maxd':
    order_index = np.argsort(np.max(centroids_dist,axis=1)) 
elif method == 'mind':
    order_index = np.argsort(np.min(centroids_dist,axis=1))
else:
    order_index = np.random.permutation(range(nrow))

labels = kmeans.labels_.astype(np.int)

for i in order_index:
    bestcl = np.argmin(centroids_dist[i,])
    labs[i] = bestcl
    clsizes[bestcl] = clsizes[bestcl] + 1
    if clsizes[bestcl] >= clsize: centroids_dist[:,bestcl] = np.inf

labs

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
y_pred = labs.astype(np.int)
plt.title('kmeans', size=18)
colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))

colors = np.append(colors, ["#000000"])
plt.scatter(data[:, 0], data[:, 1], s=10, color=colors[y_pred])