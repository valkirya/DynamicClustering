# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
from itertools import cycle, islice
import numpy as np
import folium

class Plots():
    
    cols_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
        '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
        '#000075', '#808080']
    
    brazil_location = [-10.3333333, -53.2]
    
    def curve_best_num_cluster (method, data, begin, k):
        data = np.array(data)
        fig, ax = plt.subplots()
        ax.plot(range(begin, begin+len(data)), data)
        ax.axvline(k, ls='--', color="red", label="k = "+ str(k))
        ax.set(title=str(method), xlabel = "Number of clusters", ylabel= "Distortion")
        ax.legend()
        ax.grid(True)
        plt.show()
        
    def cluster_view (method, data, labels, centers = None):
        data = np.array(data)
        fig, ax = plt.subplots()
        plt.title(str(method))
        colors = np.array(list(islice(cycle(Plots.cols_list),int(max(labels) + 1))))
        plt.scatter(data[:, 0], data[:, 1], s=10, color=colors[labels])
        
        if centers is not None: plt.scatter(centers[:,0], centers[:,1], color='black')
        
        plt.show()
        
    def data_view (service_name, data):
        data = np.array(data)
        plt.scatter(data[:,0], data[:,1], marker='.')
        plt.title(str(service_name), size=18)
        plt.xlabel('Latitude')  
        plt.ylabel('Longitude')  
        plt.show()
        
    def data_iteractive_view (data, name):
        ## initialize the map with the starting location   
        maps = folium.Map(location = [data['Lat'].mean(), data['Long'].mean()], zoom_start=12)

        ## add points
        data.apply(lambda row: folium.CircleMarker(location=[row["Lat"],row['Long']], popup ='Lat: {}, Long: {}'.format(row["Lat"], row['Long']), color='#1787FE',             fill=True, radius=0.4).add_to(maps), axis=1)
        
        ## plot the map    
        maps.save('data_'+name+'.html')
    
 
    def cluster_iteractive_view(data, labels, sizes, centers = None):
        
        colors = np.array(list(islice(cycle(Plots.cols_list), int(max(labels) + 1))))
        m = folium.Map(location = [data['Lat'].mean(), data['Long'].mean()], zoom_start=12)
        
        data.apply(lambda row: folium.CircleMarker(
            location = [row["Lat"], row['Long']], 
            popup ='Lat:{}, Long:{}, Cluster:{}, ClusterSize:{}, Index:{}'.format(row["Lat"], row['Long'], labels[row.name], sizes[labels[row.name]], row.name),
            tooltip = labels[row.name],
            color = colors[labels[row.name]],
            fill = True,
            radius = 0.4).add_to(m), axis=1)
        
        if centers is not None:
            colour = '#000000' 
            for x in centers:
                folium.CircleMarker(
                    location= [x[0], x[1]],
                    radius=5,
                    color=colour,
                    fill=True,
                    fill_color=colour,
                    popup ='Lat:{}, Long:{}'.format(x[0], x[1]),
                ).add_to(m) 
        
        # m.save(str(step)+'.html')   
        return m
        
    def comparing_approaches ( labels1, labels2):
        labels1.value_counts().plot.hist(bins=70, alpha=0.4, label='1')
        labels2.value_counts().plot.hist(bins=70, alpha=0.4, label='2')
        plt.legend()
        plt.title('Comparing Approaches')
        plt.xlabel('Cluster Sizes')