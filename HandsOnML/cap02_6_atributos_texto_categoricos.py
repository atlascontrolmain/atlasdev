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

ordinal_encoder = OrdinalEncoder() #NO RELEVANTE PARA ESTA RESOLUCIÓN

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)#Nvectores del trainset con ordinal encoder

cat_encoder = OneHotEncoder(sparse_output=False)#vectores trainset usando onehot --> lo transforma directo a una matriz
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
cat_encoder.categories_

cat_encoder.feature_names_in_
cat_encoder.get_feature_names_out()
cat_encoder.handle_unknown = "ignore"

df_output = pd.DataFrame(cat_encoder.transform(housing_cat),
                         columns=cat_encoder.get_feature_names_out(),
                         index=housing_cat.index)
print(df_output)