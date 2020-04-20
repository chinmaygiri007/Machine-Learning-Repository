#Implementation of Grid Search
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

#Importing Support_Vector_Classifier and training the model
from sklearn.svm import SVC
#classifier = SVC(kernel="linear",random_state = 0)
classifier = SVC(kernel = "rbf",random_state = 0,gamma = "auto")
classifier.fit(X_train,Y_train) 

#Predicting the data
Y_pred = classifier.predict(X_test)


#Implementing K_fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,X = X_train, y = Y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_	

#Calculating the Accuracy of the Model	
from sklearn.metrics import confusion_matrix,accuracy_score
print("Accuracy Score of SVM classifier",accuracy_score(Y_test,Y_pred)*100)

#visualization the data(Training Data)
'''
from matplotlib.colors import ListedColormap
X_set,Y_set = X_train,Y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step = 0.01),
                    np.arange(start = X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step = 0.1))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.75,cmap = ListedColormap(("black","white")))

plt.xlim(X1.min(),X1.max()) 
plt.ylim(X2.min(),X2.max())

for i,j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0],
X_set[Y_set == j,1],c=ListedColormap(("gray","blue"))(i),label = j)
'''


#Visualizing the data(Test data)
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
