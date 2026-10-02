import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Descarga de datos
data_root = "https://github.com/ageron/data/tree/main/lifesat"
lifesat = pd.read_csv (data_root + "https://raw.githubusercontent.com/ageron/data/refs/heads/main/lifesat/lifesat_full.csv")
x = lifesat [['GDP per capita (usd)']].values
y = lifesat [['Life satisfaction']].values

# Visualizar datos
lifesat.plot (kind='scatter', grid=True, 
              x = "GDP per capita (USD)",
              y = "Life satisfaction")
plt.axis ([23_500, 62_500, 4, 9])
plt.show()

# Seleccionar modelo de regresión lineal
model = LinearRegression()

# Entrenamiento de modelo
model.fit(x, y)

X_new = [[37_655.2]]
print (model.predict(X_new))