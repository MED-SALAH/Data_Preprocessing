# Regression lineaire simple

# importer les librairies 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values # varialble indépondante
y = dataset.iloc[:, -1].values # variable déponddante

# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1.0/3, random_state = 0)

# Construction du Modéle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Faire de nouvelle Prédictions
y_pred = regressor.predict(x_test)
regressor.predict(15)

# Visualiser les résultat
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salaire vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salaire')
plt.show()