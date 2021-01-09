# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:06:07 2021

@author: lcota
"""

from ipyleaflet import Map, Marker, basemaps

import matplotlib.pyplot as plt
from itertools import cycle, islice
import numpy as np
import folium

cols_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
    '#000075', '#808080']

    
colors = np.array(list(islice(cycle(cols_list), int(max(labels) + 1))))
m = Map(center = [data['Lat'].mean(), data['Long'].mean()], zoom=5, basemap=basemaps.OpenStreetMap.Mapnik)

for _, row in data.iterrows():
   
    marker = Marker(
    location = [row["Lat"], row['Long']], draggable=False,
    title ='Lat:{}, Long:{}, Cluster:{}, ClusterSize:{}, Index:{}'.format(row["Lat"], row['Long'], labels[row.name], sizes[labels[row.name]], row.name),
    #tooltip = labels[row.name],
    #color = colors[labels[row.name]]
    )
    m.add_layer(marker)

m.save('my.html', title='My Map')
