#Implementing K-means Clustering Algorithm to separate the data into various Classes

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the data
data = pd.read_csv("mall.csv")

X = data[["Annual Income (k$)","Spending Score (1-100)"]].values

#Implementing Elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++",max_iter = 300,n_init = 10,random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#Visualizing the Elbow to find out the exact prediction of No. of Clusters to be Form.
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS")
plt.show()

#Applying K-means to the Mall dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5,init = "k-means++",max_iter = 300,n_init = 10,random_state = 0)
Y_means = kmeans.fit_predict(X)

#Visualizing the Clusters
plt.scatter(X[Y_means == 0,0],X[Y_means == 0,1],s = 100,color = "red",label = "Cluster 1")
plt.scatter(X[Y_means == 1,0],X[Y_means == 1,1],s = 100,color = "blue",label = "Cluster 2")
plt.scatter(X[Y_means == 2,0],X[Y_means == 2,1],s = 100,color = "cyan",label = "Cluster 3")
plt.scatter(X[Y_means == 3,0],X[Y_means == 3,1],s = 100,color = "green",label = "Cluster 4")
plt.scatter(X[Y_means == 4,0],X[Y_means == 4,1],s = 100,color = "magenta",label = "Cluster 5")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s = 300,c = "Yellow",label = "centroid")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()

