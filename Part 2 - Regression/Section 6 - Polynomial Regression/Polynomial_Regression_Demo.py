#Creating the model using Polynomial Regression and fit the data and Visualize the data.
#Check out the difference between Linear and Polynomial Regression

#Import required libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

#Reading the data
data = pd.read_csv("Position_Salaries.csv")

X = data.iloc[:,1].values.reshape(-1,1)
Y = data.iloc[:,2].values

#Since the data is too small no need to Split the data.
#Importing Linear Regression and training the model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Visualizing the data(Linearly)
plt.scatter(X,Y,color="gray")
plt.plot(X,lin_reg.predict(X),color = "black")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.show()

#Implement Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_reg = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_reg, Y)

#Visualizing the data(Poly)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color = "black")
plt.plot(X_grid,pol_reg.predict(poly_reg.fit_transform(X_grid)),color="black")
plt.show()

#Testing the Model
print(pol_reg.predict(poly_reg.fit_transform([[5.5]]).reshape(1,-1)))
