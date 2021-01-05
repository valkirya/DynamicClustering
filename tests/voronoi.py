import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mat = np.random.rand(10,2) #random
from scipy.spatial import Voronoi, voronoi_plot_2d

vor = Voronoi(mat)
fig = voronoi_plot_2d(vor)
ridge_points = vor.ridge_points

#substituir pela info de cluster
ridge_clusters = pd.DataFrame(ridge_points)

ridge_clusters.drop_duplicates(inplace=True, ignore_index = True)
neighborh_cluster = {i: np.append(ridge_clusters[ridge_clusters[0] == i][1],ridge_clusters[ridge_clusters[1] == i][0]) for i in range(21)}
