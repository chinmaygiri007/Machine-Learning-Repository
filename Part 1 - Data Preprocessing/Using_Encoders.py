#Encoders are used to Convert Categorical values to Numerical values
#Using two types of encoders
#1.LabelEncoder 2.OneHotEncoder


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


#Converting Catergorical values to Numerical using Label_Encoder
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

#Converting Catergorical values to Numerical using One_Hot_Encoder
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X1 = onehotencoder.fit_transform(X).toarray()
