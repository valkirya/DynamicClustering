# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 19:13:48 2020

@author: lcota
"""

from k_means_constrained import KMeansConstrained

clf = KMeansConstrained(
     n_clusters=2,
    size_min=2,
    size_max=5,
    random_state=0 )
clf.fit(X)

clf.cluster_centers_
clf.predict([[0, 0], [4, 4]])
