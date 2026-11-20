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

# Transformar los datos de ocean_proximity a valores
housing_cat=housing[['ocean_proximity']] #saca vectores "no traducidos"

cat_encoder = OneHotEncoder(sparse_output=False)#vectores trainset usando onehot --> lo transforma directo a una matriz
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
cat_encoder.categories_

cat_encoder.feature_names_in_
cat_encoder.get_feature_names_out()
cat_encoder.handle_unknown = "ignore"

df_output = pd.DataFrame(cat_encoder.transform(housing_cat),
                         columns=cat_encoder.get_feature_names_out(),
                         index=housing_cat.index)

min_max_scaler = MinMaxScaler(feature_range=(-1, 1)) #sin uso de momento
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())#Escalado de datos 
#usando standardscaler porque es menos sensible a outliers

print(len(scaled_labels), 
      'es el total de distritos diferentes en el train set') #todos los distritos del trainset 
#pero solo el tamaño

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels) # ajustando datos de ingreso 
#medio por distrito escalados con standardscaler

some_new_data = housing[["median_income"]].iloc[:5] # 5 distritos con su respectivo "ingreso medio $"
# pretendemos pensar que son datos nuevos (esto es una simulación de predicción de precio medio de casa)
# partir de la median_income
print(some_new_data)#apoyo visual

scaled_predictions = model.predict(some_new_data)#retorno del precio medio 
#de las casas de los 5 distritos pero escalados aun no en usd
print(scaled_predictions)#apoyo visual

predictions = target_scaler.inverse_transform(scaled_predictions)#transformación de valores escalados a dolares
print('5 distritos, ' \
'cada valor es el precio medio de las casas por distrito. ' \
'Predicción usando median_income',predictions)
