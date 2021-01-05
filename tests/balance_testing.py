
"""
I would like to cluster points into groups bounded by size. 
The two aspects that are important here are:

1. The cluster size distribution (or the deviation from the desired cluster size).
1. The quality of the clusters (i.e. how similar are points within a cluster).

In addition to the typical hierarchical clustering approach, I will test the
following iterative approaches:

1. Iterative dichotomy: large clusters are split in two until around the desired 
size (using hierarchical clustering).
1. Iterative nearest neighbor: a point and its closest neighboring points are 
assigned to a cluster and
 removed before processing another point.
1. Iterative hierarchical clustering: keeping the first cluster of the desired 
size at the "bottom" of the dendrogram.
    

### Iterative nearest neighbor

While there are more than SIZE unassigned points:

1. A point is selected. Randomly or following a rule (see below).
1. The $s-1$ closest points are found and assigned to a new cluster.
1. These points are removed.

If the total number of points is not a multiple of $s$, the remaining points 
could be either assigned to their own clusters or to an existing cluster.
Actually, we completely control the cluster sizes here so we could fix the size 
of some clusters to $s+1$ beforehand to avoid leftovers and ensure balanced sizes.

In the first step, a point is selected.
I'll start by choosing a point randomly (out of the unassigned points).
Eventually I could try picking the points with close neighbors, or the opposite, 
far from other points.
I'll use the mean distance between a point and the others to define the order at 
which points are processed.

"""

import numpy as np
from math import ceil, dist
import pandas  as pd
import collections
from scipy.spatial import distance_matrix
from scipy import cluster
import operator 
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from itertools import cycle, islice

def iterative_nearest_neighbor (data, clsize = 1500):
    nrow = data.shape[0]
    clsize_rle = pd.cut(list(range(nrow)), ceil(nrow/clsize), labels=False)
    clsize = collections.Counter(clsize_rle) 
    #Pre definição dos tamanhos dos clusters
    
    lab = np.array([None]*nrow)
    dmat = np.array(distance_matrix(data, data))
    cpt = 0
    
    while None in lab:
        lab_ii = np.isnan(lab.astype(float))
        dmat_m = dmat[lab_ii][:, lab_ii]
        ii = np.argmax(dmat_m.sum(axis=1))
        
        lab_m = np.array([None]*sum(lab_ii))
        lab_m[np.argsort(dmat_m[ii,])[:clsize[cpt]]] = cpt
        lab[lab_ii] = lab_m
        cpt = cpt + 1

    return lab

"""
While there are more than $s$ unassigned points:

1. A hierarchical clustering is built.
1. The tree is cut at increasing level until one cluster is greater than s$.
1. Assign these points to a cluster and repeat.

Instead of working at the level of the point, the idea is to find the best 
*cluster* at each step.
The hierarchical clustering integrates information across all the (available) 
points which might 
be more robust than ad-hoc rules (e.g. nearest neighbors approach).
"""

def iterative_hierarchical_clustering (data, clsize = 10):
        
    # pre definição dos tamanhos dos clusters
    nrow = data.shape[0]
    clsize_rle = pd.cut(list(range(nrow)), ceil(nrow/clsize), labels=False)
    clsizes = list(collections.Counter(clsize_rle).values())
    
    # auxiliares
    cpt = 0
    lab = np.array([None]*nrow)
    
    t0 = time.time()
    for clss in clsizes[:-1]:
        lab_ii = np.array([i for i,x in enumerate(lab) if x == None])
        dfilter = np.take(data, lab_ii, axis=0)
                
        hc_0 = cluster.hierarchy.ward(pdist(dfilter))   
        clt_size = 0
        ct = len(lab_ii) - clss
        
        while clt_size < clss:
          cluster_temp = cluster.hierarchy.cut_tree(hc_0, n_clusters=ct).flatten()
          clt_list = collections.Counter(cluster_temp)
          clt_size = max(clt_list.values())
          ct = ct - 1
    
        key = max(clt_list.items(), key=operator.itemgetter(1))[0]
        cl_sel = [i for i,x in enumerate(cluster_temp) if x ==key]
        
        lab[lab_ii[cl_sel[:clss]]] = cpt
        cpt = cpt + 1
     
    lab[np.isnan(lab.astype(float))] = cpt
    t_batch = time.time() - t0
    print(t_batch)

    return lab

###  
    labels = lab.astype("int")
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()
    plt.title('Nearest Neighbor', size=18)
    
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(labels) + 1))))
    
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[labels])

####