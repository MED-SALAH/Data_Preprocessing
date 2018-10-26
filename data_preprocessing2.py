# Data processing

# importer les librairies 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values # varialble indépondante
y = dataset.iloc[:, -1].values # variable déponddante

# Gérer les données manquantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NAN', strategy = 'mean', axis = 0)