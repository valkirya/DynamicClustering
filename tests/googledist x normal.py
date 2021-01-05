# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:24:43 2020

@author: lcota
"""
from sklearn.metrics.pairwise import haversine_distances, manhattan_distances
from geopy.distance import geodesic
from geopy.distance import great_circle
import geopy.distance

small_cls_data = np.array(data[cluster_labels==12])  
small_cls_data1 = np.array(data[cluster_labels==19])  
                    
point1 = np.array([[-23.5819],[-46.5715]])
point2= np.array([[-23.5826],[-46.5688]])
centroide12 = np.array(small_cls_data.mean(axis=0))
centroide19 = np.array(small_cls_data1.mean(axis=0))

(pairwise_distances(point1.reshape(1,2), centroide12.reshape(1,2))/
pairwise_distances(point1.reshape(1,2), centroide19.reshape(1,2)))


haversine_distances(point1.reshape(1,2), centroide12.reshape(1,2))
haversine_distances(point1.reshape(1,2), centroide19.reshape(1,2))

manhattan_distances(point1.reshape(1,2), centroide12.reshape(1,2))
manhattan_distances(point1.reshape(1,2), centroide19.reshape(1,2))

geodesic(point1.reshape(1,2), centroide12.reshape(1,2))
geodesic(point1.reshape(1,2), centroide19.reshape(1,2))

great_circle(point1.reshape(1,2), centroide12.reshape(1,2))
great_circle(point1.reshape(1,2), centroide19.reshape(1,2))

geopy.distance.vincenty(point1.reshape(1,2), centroide12.reshape(1,2))
geopy.distance.vincenty(point1.reshape(1,2), centroide19.reshape(1,2))


import googlemaps

API_key = 'AIzaSyA6cPBM4S7Y6MvYOBU4rcOQnc5T3f7Hzvc'#enter Google Maps API key
#API_key = 'AIzaSyBrAFgYvKRcMGnZzC_F2cnE10zF2Pnt8ws'

gmaps = googlemaps.Client(key=API_key)

value = gmaps.distance_matrix(point1.reshape(1,2), point2.reshape(1,2), mode='driving')
value['rows'][0]['elements'][0]['duration']['value']
gmaps.distance_matrix(point1.reshape(1,2), centroide19.reshape(1,2), mode='driving')
                       
import networkx as nx
import osmnx as ox
#If you are pip installing OSMnx, install geopandas and rtree first. It’s easiest to use conda-forge to get these dependencies installed
nx.shortest_path_length(G, point1.reshape(1,2), centroide12.reshape(1,2), weight='length')


from routingpy import MapboxValhalla
client = MapboxValhalla(api_key='mapbox_key')

client.directions(locations=point1.reshape(1,2), profile='pedestrian')


point3 = np.array([[-23.5807],[-46.5712]])
centroide5 = cluster_centers[5]
pairwise_distances(point1.reshape(1,2), centroide12.reshape(1,2))
pairwise_distances(point1.reshape(1,2), centroide19.reshape(1,2))


 def final_ajustament_google (self, cluster_labels, cluster_sizes):
        
        cluster_centers_lat_long = list(map( lambda x : self.calculate_geo_cluster_center(cluster_labels, x), list(range(self.num_cluster))))
        
        centroids_dist = pairwise_distances(self.geo_data, cluster_centers_lat_long)
        cluster_centers_lat_long = pd.DataFrame(cluster_centers_lat_long)
        
        ajuste_size = int(centroids_dist.shape[0] * 0.01)
        order_index = np.argsort(np.min(centroids_dist,axis=1)) 
        final_elements = order_index[(centroids_dist.shape[0]-ajuste_size):centroids_dist.shape[0]]
        
        filter_data = pd.DataFrame(self.geo_data.iloc[final_elements])       
        centroids_dist_min = pairwise_distances_argmin(filter_data, cluster_centers_lat_long)

        
        
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
                

    small = []
    
    for (i1, row1) in filter_data.iterrows():
        lista = []
        for (i2, row2) in  cluster_centers_lat_long.iterrows():

            LatOrigin = row1['LAT'] 
            LongOrigin = row1['LONG']
            origins = (LatOrigin,LongOrigin)
      
            LatDest = row2[0]  
            LongDest = row2[1]
            destination = (LatDest,LongDest)

            result = gmaps.distance_matrix(origins, destination, mode='driving')["rows"][0]["elements"][0]["distance"]["value"]

            lista.append(result)                
        
        small.append(np.argmin(lista))
            
    teste = np.array(small)        
    
