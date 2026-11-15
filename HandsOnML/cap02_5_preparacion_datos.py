from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix # otra forma de obtener coef. co. (r)
from sklearn.impute import SimpleImputer

def load_housing_data():
        tarball_path = Path("datasets/housing.tgz")
        if not tarball_path.is_file():
            Path("datasets").mkdir(parents=True, exist_ok=True)
            url = "https://github.com/ageron/data/raw/main/housing.tgz"
            urllib.request.urlretrieve(url, tarball_path)
            with tarfile.open(tarball_path) as housing_tarball:
                housing_tarball.extractall(path="datasets")
        return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

# HACIENDO DIVISIÓN PARA MUESTRÉO ESTRATIFICADO
# CREAMOS UNA NUEVA COLUMNA con NUEVOS INTERVALOS
housing['income_cat'] = pd.cut(housing['median_income'],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1,2,3,4,5])

#una sola división, en una línea. Es lo que usarás normalmente para separar tu test set final.
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

#eliminamos columna de income_cat y hacemos una copia de los datos originales
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
housing=strat_train_set.copy()

#Solo trabajaremos con el train set para evitar data snooping bias

#Revisando r de train set usando corr()
#cuánto correlaciona cada atributo con el valor mediano de la vivienda
corr_matrix=housing.corr(numeric_only=True)
print(corr_matrix['median_house_value'].sort_values(ascending=False))

housing=strat_train_set.drop('median_house_value',axis=1)
housing_labels=strat_train_set['median_house_value'].copy()

#[]
#Data cleaning -> 1)elimina distrito con faltante, 2)elimina atributo completo, 
# 3)reemplaza celda vacia por un valor derivado de la misma columna

#housing.dropna(subset=["total_bedrooms"], inplace=True)    # option 1

#housing.drop("total_bedrooms", axis=1)       # option 2

#median = housing["total_bedrooms"].median()  # option 3
#housing["total_bedrooms"].fillna(median, inplace=True)
#[]

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number]) #elimina ocean_proximity
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
print(housing_tr.head())