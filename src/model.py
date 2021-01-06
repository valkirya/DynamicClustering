import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin

#import multiprocessing as pool
#from osrm import call_request
#from math import ceil

class Clusters():
    
    def __init__(self, model_data, geo_data, num_cluster, parameters):
        
        self.model_data = model_data
        self.geo_data = geo_data
        
        self.num_cluster = num_cluster
        self.cluster_max_size = parameters.cluster_max_size
        self.cluster_min_size = parameters.cluster_min_size
        self.param = parameters
        
        self.voronoi_points = self.get_points_neighbors() 
        self.matrix_distance = []
        self.matrix_duration = []
           
    def clustering_algorithm (self):
        
        if self.param.type == "Disperso":

            ## connectivity matrix for structured Ward
            connectivity = kneighbors_graph(self.model_data,
                                            n_neighbors = self.param.ward_num_neighbors,
                                            include_self = False) 
            
            algo = AgglomerativeClustering(n_clusters = self.num_cluster,
                                           linkage ='ward',
                                           connectivity = connectivity).fit(self.model_data) 
        
        else: 
            
            algo = KMeans(n_clusters = self.num_cluster,
                   init = 'k-means++',
                   n_init = self.param.kmeans_num_initialization,
                   tol = self.param.kmeans_tolerance,
                   random_state = 0).fit(self.model_data)      
        
        labels = algo.labels_.astype(int)
        sizes = pd.Series(labels).value_counts()
        algoName = str(type(algo).__name__)
                
        return (labels, sizes , algoName)
        
    def check_balance_violation (self, cls_sizes ):     
        over_points = sum([max(0, x-self.cluster_max_size) for x in cls_sizes])
        viol = True if over_points>0 else False
        
        return viol
    
    def check_aglomerative_violation (self, cls_sizes ):     
        under_points = sum([max(0, self.cluster_min_size-x) for x in cls_sizes])
        viol = True if under_points>0 else False
        
        return viol
    
    def aglomerative_clusters (self, cluster_labels, cluster_sizes):        
       
        labels = cluster_labels.copy()
        sizes = cluster_sizes.copy()
        grouped_clusters = sizes[sizes < self.cluster_min_size]
        violate_clusters = sizes[sizes < self.cluster_min_size]
        
        while len(violate_clusters) > 0:
            
            clst = violate_clusters.index[0]
            neighbors_clusters = self.get_clusters_neighbors (labels)
            
            centers = np.array(list(map( lambda x : self.calculate_geo_cluster_center(labels, x), range(self.num_cluster))))
            closest_neighs = self.get_closest_cluster (clst, neighbors_clusters, sizes, centers, N=1)
            
            labels[labels==clst] = int(closest_neighs)
            self.num_cluster -= 1    
            labels[labels==self.num_cluster] = int(clst)
                  
            sizes = pd.Series(labels).value_counts()
            violate_clusters = sizes[sizes < self.cluster_min_size]                   
                                 
        return (labels, sizes, grouped_clusters)
        
    def balance_clusters (self, cluster_labels, cluster_sizes):       
       
        clustersNames = []
        labels = cluster_labels.copy()
        sizes = cluster_sizes.copy()
        centers = np.array(list(map( lambda x : self.calculate_geo_cluster_center(labels, x), range(self.num_cluster))))
        neighbors_clusters = self.get_clusters_neighbors (labels)
        
        violate_clusters = sizes[sizes > self.cluster_max_size ]

        while len(violate_clusters) > 0 :

            clst = violate_clusters.index[0]
            size = violate_clusters.values[0]
            clustersNames.append(clst)
            
            while size > self.cluster_max_size  :
                
                closest_neighs = self.get_closest_cluster (clst, neighbors_clusters, sizes, centers)
                
                # From N closest get the smaller
                neigh_sizes = sizes[closest_neighs]
                smaller_cl = closest_neighs[neigh_sizes.argmin()]
                center_smaller_cl = centers[smaller_cl]
                 
                # calculate distance to smaller center 
                cls_data = np.array(self.geo_data[labels==clst])                 
                centroids_dist = pairwise_distances(cls_data, center_smaller_cl.reshape(1, 2))
                
                # get the point closest to smaller cluster
                index_closest_point = centroids_dist.argmin()
                
                # assign this point to new cluster -> smaller
                labels[(self.geo_data.Lat == cls_data[index_closest_point][0]) & (self.geo_data.Long == cls_data[index_closest_point][1])] = smaller_cl
                
                # update centroide smmaller e cls
                new_center = self.calculate_geo_cluster_center(labels, smaller_cl)
                centers[smaller_cl] = new_center 
                
                new_center = np.array(cls_data.mean(axis=0))
                centers[clst] = new_center 
                
                # recalculate cluster neighbors                           
                neighbors_clusters = self.get_clusters_neighbors (labels)   
                  
                # recalculate actual size
                sizes = pd.Series(labels).value_counts()
                size = sizes[clst] 
                
            # check violation
            violate_clusters = sizes[sizes > self.cluster_max_size ]
        
        clustersList = np.unique(clustersNames)
                          
        return (labels, sizes, clustersList)
    
    def get_points_neighbors (self, plot=False):        
        vor = Voronoi(self.geo_data)
        if plot : voronoi_plot_2d(vor, show_vertices = False, point_size = 0.4)

        return vor.ridge_points
                
    def get_clusters_neighbors (self, cluster_labels ):      
        ridge_clusters = list(map(lambda x, y : [cluster_labels[x], cluster_labels[y]], self.voronoi_points[:,0], self.voronoi_points[:,1]))           
        ridge_clusters_df =  pd.DataFrame(ridge_clusters)
        ridge_clusters_df.drop_duplicates(inplace=True)  
                
        neighbors_clusters = {i: list(set(ridge_clusters_df[ridge_clusters_df[1]==i][0]) | set(ridge_clusters_df[ridge_clusters_df[0]==i][1])) for i in range(self.num_cluster)}
        
        return neighbors_clusters
    
    def get_closest_cluster (self, clst, neighbors_clusters, sizes, centers, N = None):
        
        if N is None : N = self.param.balance_nun_closest_neighbors
        
        # remove self from neighbors
        neighs = neighbors_clusters[clst]
        if clst in neighs : neighs.remove(clst)
        neighs = np.array(neighs)
        
        # filter neighs with size less than max_size
        filter_neighs = neighs[sizes[neighs] < self.cluster_max_size]
        
        # get N closest neighs
        neighs_distance = filter_neighs[np.argsort(pairwise_distances(centers[filter_neighs], centers[clst].reshape(1, 2)).flatten())]   
        closest_neighs = neighs_distance [:min(len(filter_neighs), N)]
        
        return closest_neighs                 
        
    def calculate_cluster_center (self, cluster_labels, clst):       
        small_cls_data = np.array(self.model_data[cluster_labels==clst])      
        return list(small_cls_data.mean(axis=0))
    
    def calculate_geo_cluster_center (self, cluster_labels, clst):
        small_cls_data = np.array(self.geo_data[cluster_labels==clst])        
        return list(small_cls_data.mean(axis=0))         
            
    def simple_center_adjustment (self, cluster_labels, cluster_sizes ):
        
        labels = cluster_labels.copy()
        sizes = cluster_sizes.copy()
        
        centers = list(map( lambda x : self.calculate_geo_cluster_center(cluster_labels, x), list(range(self.num_cluster))))      
        centroids_dist_min = pairwise_distances_argmin(self.geo_data, centers)  
        
        filtering = [list(np.where(labels == x)[0]) for x in cluster_sizes.index]
        order_index = [ item for elem in filtering for item in elem]
        
        for i in order_index:
            bestcl = centroids_dist_min[i].astype(int)
            actualcl = labels[i].astype(int)

            if actualcl != bestcl and sizes[bestcl] < self.cluster_max_size:               
                sizes[bestcl] += 1
                sizes[actualcl] -= 1
                labels[i] = bestcl
                
        return labels, sizes
            
    def real_distance_adjustement (self, cluster_labels, cluster_sizes ):
        
        centers = list(map( lambda x : self.calculate_geo_cluster_center(cluster_labels, x), range(self.num_cluster)))
        self.matrix_distance, self.matrix_duration = self.get_osrm_distances (centers)
        
        centroids_dist_min = np.argmin(self.matrix_duration,axis=1)
        sort_index = np.argsort(np.min(self.matrix_duration , axis=1))[::-1]     
                
        labels = cluster_labels.copy()
        sizes = np.array(cluster_sizes.sort_index())
        order_labels = labels[sort_index]
        
        teste = [list(np.where(order_labels == x)[0]) for x in cluster_sizes.index]
        distance_index = [ item for elem in teste for item in elem]       
        
        for i in distance_index:
            bestcl = centroids_dist_min[i].astype(int)
            actualcl = labels[i].astype(int)   
                
            if actualcl!=bestcl and sizes[bestcl]<self.cluster_max_size :
                print(actualcl, "-> ", bestcl)
                print(self.geo_data.iloc[i])
                print(self.matrix_distance[i])
                print(self.matrix_distance[i][actualcl], self.matrix_distance[i][bestcl])
                print((self.matrix_distance[i][actualcl]- self.matrix_distance[i][bestcl])/self.matrix_distance[i][actualcl])
                sizes[bestcl] += 1
                sizes[actualcl] -= 1
                labels[i] = bestcl
                
        return labels, sizes
        
    # def get_osrm_distances (self, centers):
        
    #     # convert list of list in list of tuples
    #     centers_data = list(map(tuple, centers)) 
    #     points_data = list(map(tuple, self.geo_data.to_numpy())) 
        
    #     totalPoints = len(points_data)
    #     nPointsPerRequest = ceil(self.param.osrm_request_limit/self.num_cluster)
        
    #     # particionando data em blocos para cada requisição
    #     subdata = [centers_data + points_data[x:x+nPointsPerRequest] for x in range(0, totalPoints, nPointsPerRequest)]       
    #     parameters = [(subdata[x], self.num_cluster) for x in range(len(subdata))]
        
    #     centersDist = []
    #     centersDura = []
        
    #     # normal call
    #     aux = 0
    #     for x in parameters:
    #         print(aux)
    #         dataDist, dataDura = call_request(x[0], x[1])
    #         centersDist.extend(dataDist)
    #         centersDura.extend(dataDura)
    #         aux +=1 
            
    #     #parallel call
    #     # p = pool.Pool(pool.cpu_count())
    #     # for dataDist, dataDura in p.starmap(call_request, parameters):
    #     #     centersDist.extend(dataDist)
    #     #     centersDura.extend(dataDura)
  
    #     # p.close()
    #     # p.join()

    #     matrixDist = np.array(centersDist)
    #     matrixDura = np.array(centersDura)
        
    #     return (matrixDist, matrixDura)

    def fine_adjustement (self, cluster_labels, cluster_sizes):
        
        labels = cluster_labels.copy()
        sizes = cluster_sizes.copy()
        centers = list(map( lambda x : self.calculate_geo_cluster_center(labels, x), range(self.num_cluster)))
                
        centroids_dist = pairwise_distances(self.geo_data, centers)
        #centroids_dist_index = np.argsort(centroids_dist, axis=1)[:,:2]        
        #distance_list = list(map(lambda x ,y: y[x[1]] - y[x[0]], centroids_dist_index, centroids_dist))
        distance_index = np.argsort(np.min(centroids_dist, axis=1))[::-1]
        
        connectivity = kneighbors_graph(self.geo_data, n_neighbors = 3, include_self = False).toarray()
        conn_df = pd.DataFrame(connectivity)
        df = conn_df * (labels+1) 
                
        for i in distance_index:
            
            clt_neighbors = [int(x)-1 for x in df.iloc[i] if x>0 ]
            num_neighbors = len(np.unique(clt_neighbors))
            actual = labels[i]
            first = clt_neighbors[0]
                      
            if  num_neighbors > 1:
                #print(np.min(centroids_dist[i]))
                densed = pd.Series(clt_neighbors).value_counts().index[0]

                if densed != actual and sizes[densed] < self.cluster_max_size:
                    #print(actual, '->', densed)
                    labels[i] = densed
                    df.iloc[i] = conn_df.iloc[i]*(densed+1)
                
            elif num_neighbors == 1 and actual != first and sizes[first] < self.cluster_max_size:   
               # print(actual, '-->', first)
                labels[i] = first
                df.iloc[i] = conn_df.iloc[i]*(first+1)
          
        sizes = pd.Series(labels).value_counts()
                
        return labels, sizes        
        
    def iterative_nearest_neighbor (self, cluster_labels):
        
        labels = cluster_labels.copy()
        connectivity = kneighbors_graph(self.model_data,
                                        n_neighbors = 4,
                                        include_self = False).toarray()
        conn_df = pd.DataFrame(connectivity)
        df = conn_df * (labels+1) 
         
        a = df.apply(lambda row: row.nunique() > 2, axis=1)
        k_neighs = a[a].index.tolist()
        
        while len(k_neighs)>0:
            x = k_neighs[0]
            k_neighs.pop(0)
            
            cls_neighs = df.loc[x][df.loc[x]>0]
            cross_region = [df.loc[r][df.loc[r]>0]-1 for r in cls_neighs.index if r in k_neighs]
            cls_region = [ item for elem in cross_region for item in elem]
            cls_region.extend(cls_neighs.values-1)
            densed = pd.Series(cls_region).value_counts().index[0]
                        
            #print(labels[cls_neighs.index] , int(densed))
            labels[cls_neighs.index] = int(densed)
            #print(int(densed), labels[region])
            #labels[region] = int(densed)
            df = conn_df * (labels+1) 
            
            k_neighs = list(set(k_neighs) - set(cls_neighs.index))
          
        sizes = pd.Series(labels).value_counts()
        
        return (labels, sizes)  
    