# Regression Polynomiale
 
# importer les librairies 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values # varialble indépondante
y = dataset.iloc[:, -1].values # variable déponddante

# Construction du Modéle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
regressor = LinearRegression()
regressor.fit(x_poly, y)

# Faire de nouvelle Prédictions
regressor.predict(15)

# Visualiser les résultat
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x_poly), color = 'blue')
plt.title('Salaire vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()