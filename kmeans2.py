# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 00:15:48 2022

@author: medha
"""

import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
# seed() initializes random number generator with 0
np.random.seed(0)

# make random clusters of points by using the make_blobs class
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1,1]], cluster_std=0.9)

# scatter plot of the randomly generated data.
plt.scatter(X[:, 0], X[:, 1], marker='.')

#Initialize KMeans with parameters
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# fit the KMeans model with X
k_means.fit(X)

# labels for each point is noted
k_means_labels = k_means.labels_
k_means_labels

# cluster centers are noted
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

# plot the model

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())