#Predicting the Maximun temperature depending on Minimun temperature using Linear Regression

#Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seabornInstances
%matplotlib inline

#Reading the data
data = pd.read_csv("weather.csv")

#Visualizing the data(Basic Data)
data.plot(x="MinTemp",y="MaxTemp",style = "o")
plt.title("MaxTemp vs MinTemp")
plt.xlabel("Min Temp")
plt.ylabel("Max Temp")
plt.show()

#Visualizing Max temperature using Seaborn's distplot()
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(data["MaxTemp"])

X = data["MinTemp"].values.reshape(-1,1)
Y = data["MaxTemp"].values.reshape(-1,1)

#Splitting the data in training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=0)

#Importing Linear Regression and training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the values
y_pred = regressor.predict(X_test)

#Simple Visualization of data(Model Before Training)
plt.scatter(X, Y ,color = "black")
plt.plot(X_train,regressor.predict(X_train),color = "blue")
plt.title("Model before training")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

#Visualization of Actual data vs Predicted data(After testing)
plt.scatter(X_test,Y_test,color = "gray")
plt.plot(X_test,y_pred,color="Black")
plt.title("Model after training")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

