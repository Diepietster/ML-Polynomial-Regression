# Import the required modules
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression

import os

# Import the csv file containing the data
current_path = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(current_path, "COVID19RSA.csv"))
data.head()

print(data.info())

# Extract the required features from the csv file
data['cum_tests'] = pd.to_numeric(data['cum_tests'])
data['cum_confirmed'] = pd.to_numeric(data['cum_confirmed'])

# Convert the data to numpy arrays and reshape to correct array dimension
X = data['cum_tests'].values.reshape(-1,1)
y = data['daily_confirmed'].values.reshape(-1,1)

# Extract the training and testing data
X_train = X[:-20]
y_train = y[:-20]

X_test = X[-20:]
y_test = y[-20:]

# Set the degree of the Polynomial Regression model
quadratic_featurizer = PolynomialFeatures(degree = 5)

# Transform the training, random and testing data into a quadratic data matrix of the above degree
# Train the regression model with the training data
# Predic with the random and testing data
X_poly = quadratic_featurizer.fit_transform(X_train)
QR = LinearRegression() 
QR.fit(X_poly, y_train) 

xx = np.linspace(1, 3400000, 90)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
yy = QR.predict(xx_quadratic)

X_test_quad = quadratic_featurizer.transform(X_test)
y_test_quad = QR.predict(X_test_quad)

# Create a figure and subplot
figure = plt.figure(num=None, figsize=(18,9), dpi = 80, facecolor = 'w', edgecolor = 'k')
ax = figure.subplots()

# Plot the polynomial regression line
ax.plot(xx, yy, c='r', linestyle='--')

# Plot the training data
ax.scatter(
    X_train,
    y_train,
    c='black'
)

# Plot the testing data
ax.scatter(
    X_test,
    y_test,
    c='green'
)

# Print the R^2 and RMS score to see the goodness of the fit of the data
print(f"The R^2 score is: {r2_score(y_test_quad, y_test)}")
print(f"The RMS is: {mean_squared_error(y_test_quad, y_test)}")

# Plot the titles of the graph and axis, and remove the scientifc notation
plt.title('Daily cases decrease as number of tests increase')
plt.xlabel("Cummulative Tests")
plt.ylabel("Daily Confirmed Cases")
ax.ticklabel_format(useOffset=False, style='plain')
plt.show()