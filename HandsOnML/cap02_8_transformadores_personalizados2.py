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
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import TransformedTargetRegressor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

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

housing['income_cat'] = pd.cut(housing['median_income'],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1,2,3,4,5])

strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
housing=strat_train_set.copy()

housing=strat_train_set.drop('median_house_value',axis=1)
housing_labels=strat_train_set['median_house_value'].copy()

imputer = SimpleImputer(strategy="median") # hiperparametro
housing_num = housing.select_dtypes(include=[np.number]) #elimina ocean_proximity
imputer.fit(housing_num)
imputer.statistics_

housing_num.median().values
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())#Escalado de datos 
#usando standardscaler porque es menos sensible a outliers

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1  # Avoid division by zero for constant features
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_


# 1. fit: aprende la media y la desviación de cada columna
clone = StandardScalerClone()
clone.fit(housing_tr)

print(clone.mean_)           # media de cada columna
print(clone.scale_)          # desviación de cada columna
print(clone.n_features_in_)  # 8

# 2. transform: aplica (x - media) / desviación
housing_scaled = clone.transform(housing_tr)
print(housing_scaled.shape)                  # (16512, 8)
print(housing_scaled.mean(axis=0).round(2))  # 0 en todas las columnas
print(housing_scaled.std(axis=0).round(2))   # 1 en todas las columnas

# 3. Comprobar que da lo mismo que el StandardScaler real
real = StandardScaler().fit_transform(housing_tr)
print(np.allclose(housing_scaled, real))     # True

# 4. Lo que regalan las clases base
print(clone.get_params())                                     # {'with_mean': True}  ← BaseEstimator
housing_scaled_2 = StandardScalerClone().fit_transform(housing_tr)  # ← TransformerMixin
