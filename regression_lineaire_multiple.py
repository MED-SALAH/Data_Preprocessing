# Regression lineaire Multiple

# importer les librairies 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values # varialble indépondante
y = dataset.iloc[:, -1].values # variable déponddante

# Gérer les varialbes catégoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features= [3])
x = onehotencoder.fit_transform(x).toarray()
x = x [:, 1:]

# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Construction du Modéle
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Faire de nouvelle Prédictions
y_pred = regressor.predict(x_test)
regressor.predict(np.array ([[1, 0, 130000, 140000, 300000]]))