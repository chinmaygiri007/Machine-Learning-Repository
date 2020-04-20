#Random forest with Multiple Variable/Features
#Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Reading the data
data = pd.read_csv("winequality.csv")

#Using Multiple Variable. 
#Storing it in Single independent variable X usig 2D array
X = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].values
Y = data["quality"].values

#Splitting the data in training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#Importing Random Forest regressor and training the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300)
regressor.fit(X_train,Y_train)

#Predicting the values
y_pred = regressor.predict(X_test)

#Creating the DataFrame of Actual and Predicted data
df = pd.DataFrame({"Actual":Y_test,"Predicted":y_pred})
df1 = df.head(25)

#Visualizing the data in Bar graph
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='1', color='green')
plt.grid(which='minor', linestyle=':', linewidth='1', color='black')
plt.show()


