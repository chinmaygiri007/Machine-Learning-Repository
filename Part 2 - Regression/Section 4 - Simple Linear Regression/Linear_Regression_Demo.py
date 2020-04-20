#Predicting the Salary on the basis of Years of Experience of the employee using Simple Linear Regression

#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the data
data = pd.read_csv("Salary_Data.csv")

X = data["YearsExperience"].values.reshape(-1,1)
Y = data["Salary"].values.reshape(-1,1)

#Splitting the data in training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train , Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0,shuffle=True)

#Importing Linear Regression and training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the values
Y_pred = regressor.predict(X_test)

#Simple Visualization of data(Model Before Training)
plt.scatter(X, Y ,color = "black")
plt.plot(X_train,regressor.predict(X_train),color = "blue")
plt.title("Model before training")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

#Visualization of Actual data vs Predicted data(After testing)
plt.scatter(X_test, Y_test ,color = "black")
plt.plot(X_test,Y_pred,color = "blue")
plt.title("Model after training(While testing)")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
