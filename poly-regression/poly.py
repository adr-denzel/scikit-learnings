# reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.filter.html
# reference: https://codekarim.com/node/39
# reference: https://stackoverflow.com/questions/59829077/how-to-display-r-squared-value-on-my-graph-in-python

"""
Polynomial Regression Demo

I will be modelling a polynomial relationship
between the size of the engine (l) of an assortment
of vehicles travelling on Canadian Roads in 2022,
vs the carbon emissions (g/km) for that engine
size.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# load training data
df1 = pd.read_csv('MY2022 Fuel Consumption Ratings.csv').to_numpy()
canada_consumptions_arr = pd.array(df1)

# consumption training data
x_train = [i[4:5] for i in canada_consumptions_arr]
y_train = [i[12:13] for i in canada_consumptions_arr]

# linear model to get correlation score and compare to polynomial model
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 9, 946)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy, c='g')

# polynomial regression model
polynomial_model = PolynomialFeatures(degree=3)

# data transformation
x_train_polynomial = polynomial_model.fit_transform(x_train)

# training the model
regressor_polynomial = LinearRegression()
regressor_polynomial.fit(x_train_polynomial, y_train)
xx_quadratic = polynomial_model.transform(xx.reshape(xx.shape[0], 1))
yy_quadratic = regressor_polynomial.predict(xx_quadratic)

plt.plot(xx, yy_quadratic, c='r', linestyle='--')
plt.scatter(x_train, y_train, c='b')
plt.title('Motor Car Fuel Consumption v CO2 Emission')
plt.xlabel('Engine Size (L)')
plt.ylabel('Carbon Emissions (g/100km)')
plt.grid(True)
plt.axis([0, 9, 0, 650])
plt.show()
