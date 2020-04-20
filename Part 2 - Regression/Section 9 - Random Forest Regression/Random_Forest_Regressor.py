#Predicting the Salary based on level of Employee in company using Random Forest Regressor

#Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the data
data = pd.read_csv("Position_Salaries.csv")

X = data["Level"].values.reshape(-1,1)
Y = data["Salary"].values

#Since the data is too small no need to Split the data.
#Importing Random Forest regressor and training the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state = 0)
regressor.fit(X,Y)

#Predicting the data
Y_pred1 = regressor.predict(X)

#Visualizing the data
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color = "red")
plt.plot(X_grid, regressor.predict(X_grid),color = "black")
plt.title("Level vs Salary(Using Decision Tree Regression)") 
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()




