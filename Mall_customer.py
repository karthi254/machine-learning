# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:07:34 2023

@author: gunak
"""

import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

sns.set(context="paper", palette="Blues",style = "whitegrid",font_scale= 1, color_codes=True)
data = pd.read_csv("Mall_Customers.csv",index_col="CustomerID")

print(data.head(7))

x = data.iloc[:,[2,3]].values
print(x)
wcss = []

for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = "k-means++", random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10,5))  
sns.lineplot(range(1,20),wcss,marker="o",color="red")
plt.title("The Elbow Method")
plt.xlabel("Number of cluster")
plt.ylabel("Wcss")
plt.show()

kmeans = KMeans(n_clusters = 5, init = "k-means++",random_state = 42)
y_kmeans = kmeans.fit_predict(x)

plt.figure(figsize=(15,7))
sns.scatterplot(x[y_kmeans == 0,0], x[y_kmeans == 0, 1], color = "yellow", label = "cluster 1", s=150)
sns.scatterplot(x[y_kmeans == 1,0], x[y_kmeans == 1, 1], color = "blue", label = "cluster 2", s=150)
sns.scatterplot(x[y_kmeans == 2,0], x[y_kmeans == 2, 1], color = "green", label = "cluster 3", s=150)
sns.scatterplot(x[y_kmeans == 3,0], x[y_kmeans == 3, 1], color = "grey", label = "cluster 4", s=150)
sns.scatterplot(x[y_kmeans == 4,0], x[y_kmeans == 4, 1], color = "orange", label = "cluster 5", s=150)
sns.scatterplot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color = "red",
                label = "Centroids", s=200,marker=",")
plt.grid(False)
plt.title("Clusters of customers")
plt.xlabel("Annual income")
plt.ylabel("spending score (1 to 100)")
plt.legend()
plt.show()












