
"""
Created on Thu Oct 15 10:35:03 2020
@author: lcota
[Same-size k-Means Variation]
"""

import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import sys  
import os
sys.path.append(os.path.abspath(os.path.join('..', 'data')))

#
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances_argmin, pairwise_distances
    from sklearn.metrics.pairwise import haversine_distances, manhattan_distances
from sklearn.metrics import silhouette_score, davies_bouldin_score
from math import radians

DATA_DIRECTORY = 'data'
FILE_NAME = 'demands.csv'
ROUTE_LIMIT = 1500

# sub_cluster,vol,latit,longit,cidade
# TODO já tiver cabeçalho
df = pd.read_csv("../"+DATA_DIRECTORY+"/"+FILE_NAME, names=['SUBCLUSTER','VOL','LAT','LONG','CIDADE'])

# TODO Se não contiver vol
df = df.groupby(['LAT','LONG'], as_index=False).VOL.sum()
num_points = df.VOL.count() 

num_cluster = np.ceil(df.VOL.sum()/ROUTE_LIMIT).astype('int')
num_cluster = np.ceil(df.VOL.count()/ROUTE_LIMIT).astype('int')

# Ploting
X = pd.DataFrame(StandardScaler().fit_transform(df[['LAT','LONG']]))
X = np.array(df[['LAT','LONG']])
data = np.vectorize(radians)(X)
 
plt.scatter(X[:,0],X[:,1], marker='.')

## Defining Best K
def optimal_number_of_clusters(wcss, initial_num_cluster):
    x1, y1 = initial_num_cluster, wcss[0]
    x2, y2 = initial_num_cluster+len(wcss), wcss[-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return initial_num_cluster + distances.index(max(distances))
##
# The optimal number of clusters is somehow subjective and depends on the method used for measuring 
# similarities and the parameters used for partitioning.
# calculate the Within-Cluster-Sum of Squared Errors (WSS) for different values of k,
# and choose the k for which WSS becomes first starts to diminish. 

#Distortion: It is calculated as the average of the squared distances from the cluster centers of
# the respective clusters. Typically, the Euclidean distance metric is used.
# Inertia: It is the sum of squared distances of samples to their closest cluster center.
       
def elbow_method (data, initial_num_cluster = 0, final_num_cluster = 8):
    wcss = [] # within clusters sum of squares
    haversine_dist = []
    
    spatial_dist = []
    haversine_dist = []
    distortions = []
    euclidean_dist = []
    city_dist = []
    sil = []
    dbs = []
    
    # Scikit-learn’s KMeans already calculates the wcss and its named inertia.
    
    for n in range(initial_num_cluster, final_num_cluster+1):
        kmeans = MiniBatchKMeans(n_clusters = n, 
                                 max_iter = num_points * 10, 
                                 batch_size = 750, 
                                 max_no_improvement = 5)
        kmeans.fit(X)      
        wcss.append(kmeans.inertia_)
        haversine_dist.append(sum(np.min(haversine_distances(data, kmeans.cluster_centers_),
                                         axis=1))/X.shape[0]) 
        
        spatial_dist.append(sum(np.min(manhattan_distances(X, kmeans.cluster_centers_), axis=1))) 
        city_dist.append(sum(np.min(pairwise_distances(X, kmeans.cluster_centers_, metric='cityblock'), axis=1))) 
        
        
        #sil.append(silhouette_score(X, kmeans.labels_)
        #dbs.append(davies_bouldin_score(X, kmeans.labels_))
    
    #sil.index(max(sil)) + initial_num_cluster
    #dbs.index(min(dbs)) + initial_num_cluster
                        
    optimal_number_of_clusters(wcss, initial_num_cluster, final_num_cluster)
    optimal_number_of_clusters(haversine_dist, initial_num_cluster, final_num_cluster)
    optimal_number_of_clusters(spatial_dist, initial_num_cluster, final_num_cluster)
    optimal_number_of_clusters(city_dist, initial_num_cluster, final_num_cluster)
    
    return n

# 
# The silhouette value measures how similar a point is to its own cluster (cohesion) 
# compared to other clusters (separation).

### main 
# calculating the within clusters sum-of-squares for 19 cluster amounts
elbow_method(X, 22, 26)
  
    
# Clustering Algo Parameters

# manually fit on batches
kmeans = MiniBatchKMeans(n_clusters = num_cluster,
                         init = 'k-means++',
                         max_iter = num_points * 10,
                         batch_size  = 750,
                         compute_labels  = True,
                         max_no_improvement  = 1,
                         reassignment_ratio = 1)
t0 = time.time()
kmeans.fit(X) 
t_batch = time.time() - t0
print(t_batch)

###
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
y_pred = kmeans.labels_.astype(np.int)
plt.title('kmeans', size=18)

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))

# add black color for outliers (if any)
colors = np.append(colors, ["#000000"])

plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

# info
mbk_means_cluster_centers = np.sort(kmeans.cluster_centers_, axis = 0) 
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers) 

# clusters sizes

preds=pd.DataFrame()
preds['kmeans'] = kmeans.labels_.astype(np.int)
preds.kmeans.value_counts()

sum(preds.kmeans.value_counts() > 1500 )   

#### Abouth Neighbor Clusters
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
connect = connectivity.toarray()


# Plot 2    
from random import randint
colors = []
for i in range(num_cluster): colors.append('#%06X' % randint(0, 0xFFFFFF))

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
for k, col in zip(range(num_cluster), colors):
    my_members = mbk_means_labels == k
    cluster_center = mbk_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)


#################################################################################

def silhouette_coefficient (data, min_num_cluster, num_test, param, plot = False):
    labels = [] 
    
    for n in range (min_num_cluster, min_num_cluster+num_test):
        kmeans = KMeans(n_clusters = n,
                        init = 'k-means++',
                        n_init = param.kmeans_num_initialization,
                        tol = param.kmeans_tolerance,
                        random_state = 0).fit(data)
        labels.append(kmeans.labels_)
        
    silhouette = []
    parameters = [(data, x) for x in labels]
    
    p = multiprocessing.Pool(multiprocessing.cpu_count())
        
    for solution in p.starmap(silhouette_score, parameters):
        silhouette.append(solution)   
        
    p.close()
    p.join()

    ## best k: the large value
    k = silhouette.index(max(silhouette)) + 1
    
    if plot: Plots.curve_best_num_cluster ("The Silhouette Coefficient", silhouette, min_num_cluster + 1, min_num_cluster + k)
    
    return k
    
    
# # connectivity matrix for structured Ward
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

ward = AgglomerativeClustering( n_clusters = num_cluster, linkage='ward', connectivity=connectivity)

t0 = time.time()
ward.fit(X) 
t_batch = time.time() - t0
print(t_batch)

preds=pd.DataFrame()
preds['ward'] = ward.labels_.astype(np.int)
preds.ward.value_counts()

sum(preds.ward.value_counts() > 1500 )   

clustering_algorithms = (
   # ('MiniBatchKMeans', kmeans),
   # ('AffinityPropagation', affinity_propagation),
    #('MeanShift', ms),
    #('SpectralClustering', spectral),
    ('Ward', ward),
    #('AgglomerativeClustering', average_linkage),
    #('DBSCAN', dbscan),
    #('OPTICS', optics),
    #('Birch', birch),
    #('GaussianMixture', gmm)
)
    
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot()
y_pred = ward.labels_.astype(np.int)
plt.title('Ward', size=18)

colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                     '#f781bf', '#a65628', '#984ea3',
                                     '#999999', '#e41a1c', '#dede00']),
                              int(max(y_pred) + 1))))

# add black color for outliers (if any)
colors = np.append(colors, ["#000000"])

plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
