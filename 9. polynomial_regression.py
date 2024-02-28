import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X = X.flatten()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1) 
degree = 2
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)
sort_indices = np.argsort(X_test.flatten())
X_test_sorted = X_test[sort_indices]
y_test_sorted = y_test[sort_indices]
y_pred_sorted = y_pred[sort_indices]
plt.scatter(X_test_sorted, y_test_sorted, color='black')
plt.plot(X_test_sorted, y_pred_sorted, color='blue', linewidth=3)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression (Degree={})'.format(degree))
plt.show()
