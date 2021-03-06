#Implementing the Logistic Regression to Classify the Data in Purchased/Not Purchased Class

#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the data
data = pd.read_csv("Social_Network_Ads.csv")

X = data[["Age","EstimatedSalary"]].values
Y = data["Purchased"].values

#Splitting the data in training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)

#Applying Feature Scaling for data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Importing Logistic Regression and training the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

#Predicting the data
Y_pred = classifier.predict(X_test)

#Visualizing the data
from matplotlib.colors import ListedColormap
X_set, Y_set = X_test,Y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1,step=0.01),
                    np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape)
            ,alpha = 0.75,cmap = ListedColormap(["red","green"]))

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0],
                X_set[Y_set == j,1],
                c = ListedColormap(("blue","black"))(i),label = j)
