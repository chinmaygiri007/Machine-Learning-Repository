#Imputer is use to Fill the Missing Values in dataset Using Various Strategies like:
# 1.Mean 2.Median 3.Most Frequent

#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the data
data = pd.read_csv("Data.csv")

X = data[["Country","Age","Salary"]].values
Y = data[["Purchased"]].values

#Importing and fitting Imputer to find missing values and Insert the mean of Values
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])

#Transforming the columns
X[:,1:3] = imputer.transform(X[:,1:3])

