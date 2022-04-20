# help from here: https://towardsdatascience.com/linear-regression-using-least-squares-a4c3456e8570

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

d = load_diabetes()
d_X = d.data[:, np.newaxis, 2]
dx_train = d_X[:-20].squeeze()
dy_train = d.target[:-20]
dx_test = d_X[-20:].squeeze()
dy_test = d.target[-20:]

# y = mx + c is the function I'm working with
# c = y - mx

# finding the mean of all x and mean of all y of the training set
x_mean = np.mean(dx_train)
y_mean = np.mean(dy_train)

# following the function to divine the mean of the least squares provided in pdf
numerator = 0
denominator = 0

# getting sum of differences of values to means
for i in range(len(dx_train)):
    numerator += (dx_train[i] - x_mean) * (dy_train[i] - y_mean)
    denominator += (dx_train[i] - x_mean) ** 2

# applying formula from pdf
m = numerator / denominator
c = y_mean - m * x_mean


# define line function to apply test set to
def line(x):
    return x * m + c


# plot all scatters and lines
plt.scatter(dx_test, dy_test, c='g', label='Testing Data')
plt.scatter(dx_train, dy_train, c='r', label='Training Data')
plt.plot(dx_test, line(dx_test), c='b', label='Predicted Line')
plt.legend()
plt.show()
