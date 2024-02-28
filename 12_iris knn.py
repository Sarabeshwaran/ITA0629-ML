"""
Created on Tue Jan 24 09:37:46 2023

@author: GEOMOL GEORGE
Iris Flower Classification using Python
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris = pd.read_csv(r"C:\Users\welcome\Documents\IRIS.csv")
#first five rows of this dataset:
print(iris.head())    
print(iris.describe())
#The target labels of this dataset are present in the species column, let’s have a quick look at the target labels:

print("Target Labels", iris["species"].unique())
#plot the data using a scatter plot which will plot the iris species according to the sepal length and sepal width:

import plotly.io as io
io.renderers.default='browser'

import plotly.express as px
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()
#Iris Classification Model

x = iris.drop("species", axis=1)
y = iris["species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

x_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
"""
from sklearn import metrics
print("Prediction: {}".format(prediction))
print ("-------------------------------------------------------------------------")
print("\nConfusion Matrix:\n",metrics.confusion_matrix(y_test, ypred))  
print ("-------------------------------------------------------------------------")
print("\nClassification Report:\n",metrics.classification_report(y_test, ypred)) 
print ("-------------------------------------------------------------------------")
print('Accuracy of the classifer is %0.2f' % metrics.accuracy_score(y_test,ypred))
print ("-------------------------------------------------------------------------")
"""
