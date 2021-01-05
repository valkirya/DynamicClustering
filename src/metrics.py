from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from visualization import Plots
from math import sqrt
import numpy as np
import pandas as pd

class Metrics():
    
    def calculate_min_num_cluster (model_data, param):
        
        # find initial best num cluster
        # compromise between dense and non-dense regions
        num_cluster_by_size = np.ceil(model_data.shape[0]/param.cluster_max_size).astype(int)
        
        # A Density-Based Algorithm - DBSCAN
        # Agorithm views clusters as areas of high density separated by areas of low density.         
        clustering = DBSCAN(eps = param.dbscan_maximum_distance,
                            min_samples = param.dbscan_neighborhood_size).fit(model_data)
        
        num_cluster_by_density = max(clustering.labels_)+1
        min_num_cluster = int((num_cluster_by_size+num_cluster_by_density)/2)
        
        return int(min_num_cluster)
    
    def calculate_best_num_cluster (data, param, min_num_cluster):
        wcss, pts, labels = [] , []  , [] 
        check_balance = 0
        
        balance_threshold = param.balance_feasibility_threshold        
        min_testing = min_num_cluster + param.num_experiments
        n = min_num_cluster
        
        # num test não fixo, minimo de 10 até respeitar balance threshold 
        while n <= min_testing or check_balance == 0: 

            kmeans = KMeans(n_clusters = n,
                           init = 'k-means++',
                           n_init = param.kmeans_num_initialization,
                           tol = param.kmeans_tolerance,
                           random_state = 0).fit(data)
            
            n += 1
            p = [max(0, x-param.cluster_max_size) for x in pd.Series(kmeans.labels_).value_counts()]
            if sum(np.array(p) > balance_threshold) == 0: check_balance += 1
            
            wcss.append(kmeans.inertia_)
            labels.append(kmeans.labels_)
            pts.append(sum(p))                                
               
        k1 = Metrics.elbow_method (min_num_cluster, wcss)
        k2 = Metrics.elbow_method (min_num_cluster, pts)
        
        if pts[k1] <= balance_threshold and pts[k2] <= balance_threshold: 
            list_num_cluster = [k1, k2]
        elif pts[k1] <= balance_threshold :
            list_num_cluster = [k1]
        elif pts[k2] <= balance_threshold :
            list_num_cluster = [k2]
        else:
            pts_over = list(filter(lambda i: i < balance_threshold , pts))
            list_num_cluster = [pts.index(pts_over[0])]
                   
        if len(list_num_cluster)>1:
            silhouette = []
            
            for x in range(len(list_num_cluster)):
                if max(labels[x]) > 0: silhouette.append(silhouette_score(data, labels[x]))
                                  
            ## best k: the large value
            k = list_num_cluster[silhouette.index(max(silhouette))]
            
        else :
            k = list_num_cluster[0]
            
        best_k = min_num_cluster + k
                     
        return (best_k, n-min_num_cluster)
    
    """
    The elbow method calculates the Within-cluster-sum of Squared Errors (WSS) for different values of k,and choose the k for which WSS becomes first starts to diminish. 

    """
    
    def elbow_method (min_num_cluster, wcss, plot = False):
        x1, y1 = min_num_cluster, wcss[0]
        x2, y2 = min_num_cluster+len(wcss), wcss[-1]
    
        distances = []
        for i in range(len(wcss)):
            x0 = i+2
            y0 = wcss[i]
            a = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            b = sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(a/b)
        
        ## best k: the large distance value
        k = distances.index(max(distances))
        
        if plot: Plots.curve_best_num_cluster ("The Elbow Method", wcss, min_num_cluster, min_num_cluster + k)
        
        return k

    """
The silhouette coefficient measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation).
    """
