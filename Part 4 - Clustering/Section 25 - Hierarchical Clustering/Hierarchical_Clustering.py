#Implementing Agglomative Clustering Algorithm to separate the data into various Classes

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the data
data = pd.read_csv("mall.csv")

X = data[["Annual Income (k$)","Spending Score (1-100)"]].values

#Implementing Dendogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method = "ward"))
plt.title("Dendrogram")
plt.xlabel("Clusters")
plt.ylabel("Euclidean Distance")

#Visualizing the Elbow to find out the exact prediction of No. of Clusters to be Form.
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS")
plt.show()

#Applying Agglomative Clustering to the Mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,affinity = "euclidean",linkage = "ward")
Y_hc = hc.fit_predict(X)

#Visualizing the Clusters
plt.scatter(X[Y_hc == 0,0],X[Y_hc == 0,1],s = 100,c = "blue",label = "Cluster 1")
plt.scatter(X[Y_hc == 1,0],X[Y_hc == 1,1],s = 100,c = "yellow",label = "Cluster 2")
plt.scatter(X[Y_hc == 2,0],X[Y_hc == 2,1],s = 100,c = "pink",label = "Cluster 3")
plt.scatter(X[Y_hc == 3,0],X[Y_hc == 3,1],s = 100,c = "red",label = "Cluster 4")
plt.scatter(X[Y_hc == 4,0],X[Y_hc == 4,1],s = 100,c = "black",label = "Cluster 5")

plt.title("cluster of Customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

