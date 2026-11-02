from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
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

#haciendo spliteo de datos -> conjunto de prueba y conjunto de entrenamiento
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print (train_set,'Conjunto de ENTRENAMIENTO 0.8') #SOLO PARA PRUEBA Y VISUALIZACIÓN DE ORDEN DE DATOS
print (test_set,'Conjunto de PRUEBA 0.2') #SOLO PARA PRUEBA Y VISUALIZACIÓN DE ORDEN DE DATOS


# asignar hash y definir valor de umbral
def is_id_in_test_set(identifier, test_ratio):
     return crc32(np.int64(identifier)) < test_ratio * 2**32

# dividir nuevamente pero con hash
def split_data_with_id_hash(data, test_ratio, id_column):
     ids = data[id_column]
     in_test_set = ids.apply(lambda id_:is_id_in_test_set(id_,test_ratio))
     return data.loc[~in_test_set], data.loc[in_test_set]

# crear columna con ID
housing_with_id=housing.reset_index()

# llama al split -> apenas calcula los hashes y se separa
train_set, test_set = split_data_with_id_hash(housing_with_id,0.2,'index')
print(len(train_set),len(test_set))

# usar este forma de asignación de ID nos obliga que toda 
# la información nueva que se desee agregar vaya al final 
# del dataset y no se hagan inserciones en posiciones diferentes 
# del final --> si esto no es posible debemos cambiar el método de asignación de ID