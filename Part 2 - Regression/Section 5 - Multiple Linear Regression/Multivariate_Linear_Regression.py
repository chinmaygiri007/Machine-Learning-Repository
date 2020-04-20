#Predicting the Quality of Wine on given data using Multivarite Linear Regression

#Import required libraries
import numpy as np
import pandas as pd
import seaborn as SeabornInstance
import matplotlib.pyplot as plt 

#Reading the data
data = pd.read_csv("winequality.csv")

#Finding the missing values in Dataset
data = data.fillna("ffill")

#Using Multiple Variable. 
#Storing it in Single independent variable X usig 2D array
X = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]].values
Y = data["quality"].values

#Visualizing the quality of data
plt.figure(figsize=(15,10))
plt.tight_layout()
SeabornInstance.distplot(data["quality"])

#Splitting the data in training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)

#Importing Linear Regression and training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

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


