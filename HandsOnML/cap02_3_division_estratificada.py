from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
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
#print(housing.head())
#housing.hist(bins=50, figsize=(12, 8)) #--> sirve para saber las limitaciones con base en unidades registradas
#plt.show()
#print(housing.info())

##################################################
#SE SUSTITUYE ESTO POR EL MUESTREO ESTRATIFICADO
#haciendo spliteo de datos -> conjunto de prueba y conjunto de entrenamiento
#train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#print (train_set,'Conjunto de ENTRENAMIENTO 0.8') #SOLO PARA PRUEBA Y VISUALIZACIÓN DE ORDEN DE DATOS
#print (test_set,'Conjunto de PRUEBA 0.2') #SOLO PARA PRUEBA Y VISUALIZACIÓN DE ORDEN DE DATOS

##################################################
# ELIMINAMOS ESTA PARTE PORQUE LA DIVISIÓN CON HASH ES "ALEATORIA"
# asignar hash y definir valor de umbral
#def is_id_in_test_set(identifier, test_ratio):
#     return crc32(np.int64(identifier)) < test_ratio * 2**32

# dividir nuevamente pero con hash
#def split_data_with_id_hash(data, test_ratio, id_column):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_:is_id_in_test_set(id_,test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]

# crear columna con ID
#housing_with_id=housing.reset_index()

# llama al split -> apenas calcula los hashes y se separa
#train_set, test_set = split_data_with_id_hash(housing_with_id,0.2,'index')
##################################################


# HACIENDO DIVISIÓN PARA MUESTRÉO ESTRATIFICADO
# CREAMOS UNA NUEVA COLUMNA con NUEVOS INTERVALOS
housing['income_cat'] = pd.cut(housing['median_income'],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1,2,3,4,5])
#CONTEO DE VALORES POR DISTRITO Y ORDEMANOS LOS VALORES OBTENIDOS
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
#plt.xlabel('Income category')
#plt.ylabel('Number of districts')
#plt.show()

#DIEZ DIVISIONES, DIEZ CONJUNTOS DE PRUEBA Y ENTRANAMIENTO "ALEATORIOS" BAJO LAS CONDICIONES
#DE LAS DIVISIONES DE LOS INTERVALOS DE income_cat
splitter=StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=42)
strat_splits=[]
for train_index,test_index in splitter.split(housing, housing["income_cat"]):
     strat_train_set_n=housing.iloc[train_index]
     strat_test_set_n=housing.iloc[test_index]
     strat_splits.append([strat_train_set_n,strat_test_set_n])

#IMPRIME LA PRIMER "BARAJA" DE DIVISIONES BASADAS EN LOS INTERVALOS DE income_cat
strat_train_set, strat_test_set = strat_splits[0]
print(strat_splits[0])

#una sola división, en una línea. Es lo que usarás normalmente para separar tu test set final.
def split_estratificado_comparativo():
    strat_train_set_x, strat_test_set_y = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    return strat_train_set_x, strat_test_set_y
print (split_estratificado_comparativo())