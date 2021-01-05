# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:14:17 2020

@author: lcota
"""

    def final_adjustment_based_distance (self, cluster_labels, cluster_sizes ):
        
        ridge_points = self.get_points_neighbors(plot=False)            
        neighbors_clusters = self.get_clusters_neighbors (cluster_labels, ridge_points)     
            
        new_cluster_centers = list(map( lambda x : self.calculate_geo_cluster_center(cluster_labels, x), list(range(self.num_cluster))))
        
        radians_cluster_centers = np.vectorize(radians)(new_cluster_centers)
        centroids_dist = haversine_distances(radians_cluster_centers, radians_cluster_centers)    
       # multiply by Earth radius to get kilometers
        centroids_dist_km = centroids_dist * 6371       
                
        sizes = np.array(cluster_sizes.sort_index())
        labels = cluster_labels.copy()
        radius = 150
        
        for i in list(range(self.num_cluster)):
            actual_size = sizes[i]
            centroids_dists = centroids_dist_km[i]
            neighs = neighbors_clusters[i]
            if i in neighs : neighs.remove(i)
            neighs = np.array(neighs)
            #print(actual_size, neighs)
            
            neighs_closest = neighs[np.argsort(centroids_dists[neighs])][:2]                   
            neighs_radius_rules = centroids_dists[neighs_closest] < 150
            neighs_grouped_rules = (sizes[neighs_closest] + actual_size) < 1500
            
            ideal_grouping = neighs_closest[neighs_radius_rules & neighs_grouped_rules]
            
            if len(ideal_grouping)> 0:
                best_cls = ideal_grouping[0]
                sizes[best_cls] += actual_size
                sizes[i] = 0
                labels[labels==i] = best_cls
                
        return (labels, sizes)
    
    
    def final_adjustment_real_distance (self, cluster_labels, cluster_sizes ):
        
        new_cluster_centers = list(map( lambda x : self.calculate_geo_cluster_center(cluster_labels, x), list(range(self.num_cluster))))
        radians_cluster_centers = np.vectorize(radians)(new_cluster_centers)
        
        radians_data = np.vectorize(radians)(self.geo_data)
        centroids_dist = haversine_distances(radians_data, radians_cluster_centers)    
        order_index = np.argsort(np.min(centroids_dist,axis=1))   
        centroids_dist_min = np.argmin(centroids_dist,axis=1)
        
        labels = cluster_labels.copy()
        sizes = np.array(cluster_sizes.sort_index())
        
        for i in order_index:
            bestcl = centroids_dist_min[i].astype(int)
            actualcl = labels[i].astype(int)
            if actualcl != bestcl and sizes[bestcl] < self.cluster_max_size :
                #print(actualcl, "-> ", bestcl)
                sizes[bestcl] += 1
                sizes[actualcl] -= 1
                labels[i] = bestcl
                
        return (labels, sizes)
    
    def iterative_nearest_neighbor (self, cluster_labels):
        labels = cluster_labels.copy()
        connectivity = kneighbors_graph(self.model_data,
                                        n_neighbors = 3,
                                        include_self = False).toarray()
        conn_df = pd.DataFrame(connectivity)
        
        df = conn_df * (labels+1)    
        a = df.apply(lambda row: row.nunique() > 2, axis=1)
        k_neighs = a[a].index.tolist()
        print("Points to be balanced: ", len(k_neighs))
        for x in k_neighs:
            row = df.loc[x][df.loc[x]>0]
            new_row = row - 1
            closest = new_row.value_counts().index[0]
            labels[new_row.index] = int(closest)
            
            df = conn_df * (labels+1)    
            a = df.apply(lambda row: row.nunique() > 2, axis=1)
            k_neighs = a[a].index.tolist()
        
            #labels[new_row.index] = 11
            
        df = conn_df * (labels+1)    
        a = df.apply(lambda row: row.nunique() > 2, axis=1)
        k_neighs = a[a].index.tolist()
        print("Points to be balanced: ", len(k_neighs))
        for x in k_neighs:
            row = df.loc[x][df.loc[x]>0]
            new_row = row - 1
            closest = new_row.value_counts().index[0]
            labels[new_row.index] = int(closest)
            
            df = conn_df * (labels+1)    
            a = df.apply(lambda row: row.nunique() > 2, axis=1)
            k_neighs = a[a].index.tolist()
            
            
        sizes = pd.Series(labels).value_counts()
        #centers = list(map( lambda x : self.calculate_geo_cluster_center(labels, x), list(range(self.num_cluster))))
        centers = list(map( lambda x : self.calculate_cluster_center(labels, x), list(range(self.num_cluster))))
        
        return (labels, sizes, centers)     

    def local_search_balance ():
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.1, min_samples=2).fit(cls_data)
        cluster_labels= clustering.labels_
        pd.Series(cluster_labels).value_counts()
        
        violate_clusters = cluster_sizes[cluster_sizes > self.cluster_max_size]
        if len(violate_clusters) > 0 :
            
            vor = Voronoi(data)
            if plot : voronoi_plot_2d(vor, show_vertices= False, point_size = 0.6)
            ridge_points = vor.ridge_points
            
            ridge_clusters = list(map(lambda x, y : [cluster_labels[x], cluster_labels[y]], ridge_points[:,0], ridge_points[:,1]))           
            ridge_clusters_df =  pd.DataFrame(ridge_clusters)
            ridge_clusters_df.drop_duplicates(inplace=True)  
            
            neighbors_clusters = {i: list(set(ridge_clusters_df[ridge_clusters_df[1]==i][0]) | set(ridge_clusters_df[ridge_clusters_df[0]==i][1])) for i in range(self.num_cluster)}    
        
            # TODO paralelizar calculo
            for i in violate_clusters:
                clst = violate_clusters.index[0]
                size = violate_clusters.item()
                
                neighs = neighbors_clusters[clst]
                if clst in neighs : neighs.remove(clst)
                neigh_sizes = cluster_sizes[neighs]
            
        for cls_data = np.array(data[cluster_labels==cluster]) 
        cls_data = np.array(data[(cluster_labels==clst) | (cluster_labels==2)])                 
        centroids_dist = pairwise_distances(cls_data, center_smaller_cl.reshape(1, 2))