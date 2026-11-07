from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
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

##################################################
# VIENE DE LA PARTE 3: DIVISIÓN ESTRATIFICADA CON income_cat
housing['income_cat'] = pd.cut(housing['median_income'],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1,2,3,4,5])
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

#income_cat YA NO SE NECESITA, SOLO SIRVIÓ PARA ESTRATIFICAR -> SE ELIMINA DE AMBOS CONJUNTOS
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
##################################################


# EXPLORAR SOLO EL CONJUNTO DE ENTRENAMIENTO -> EL TEST SET NO SE TOCA
#COPIA PARA PODER EXPERIMENTAR SIN MODIFICAR EL ORIGINAL
housing = strat_train_set.copy()

##################################################
# VISUALIZAR DATOS GEOGRÁFICOS (longitud y latitud)
#SIN alpha TODOS LOS PUNTOS SE ENCIMAN Y NO SE VEN PATRONES
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
plt.show()

#CON alpha=0.2 SE RESALTAN LAS ZONAS DE ALTA DENSIDAD (Bay Area, Los Ángeles, San Diego, Central Valley)
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.show()

#RADIO DEL CÍRCULO (s) = POBLACIÓN DEL DISTRITO, COLOR (c) = PRECIO MEDIO DE LA CASA
#cmap "jet" -> AZUL = BARATO, ROJO = CARO
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()
#CONCLUSIÓN: EL PRECIO DEPENDE MUCHO DE LA UBICACIÓN (cerca del océano) Y DE LA DENSIDAD DE POBLACIÓN

##################################################
# BUSCAR CORRELACIONES
#COEFICIENTE DE PEARSON ENTRE CADA PAR DE ATRIBUTOS (solo columnas numéricas, ocean_proximity es texto)
corr_matrix = housing.corr(numeric_only=True)
#QUÉ TANTO SE CORRELACIONA CADA ATRIBUTO CON EL PRECIO MEDIO
#  cerca de 1  -> correlación positiva fuerte
#  cerca de -1 -> correlación negativa fuerte
#  cerca de 0  -> no hay correlación LINEAL (puede haber otra no lineal)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

#MATRIZ DE DISPERSIÓN: CADA ATRIBUTO CONTRA LOS DEMÁS (la diagonal muestra el histograma de cada uno)
#SOLO LOS MÁS PROMETEDORES, CON 11 ATRIBUTOS SERÍAN 121 GRÁFICAS
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

#EL ATRIBUTO MÁS PROMETEDOR ES median_income -> ZOOM A SU GRÁFICA
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.show()
#SE VEN LÍNEAS HORIZONTALES: EN $500,000 (tope de los datos) Y OTRAS EN ~$450,000, ~$350,000, ~$280,000
#ESOS DISTRITOS PODRÍAN QUITARSE PARA QUE EL MODELO NO APRENDA ESAS "PECULIARIDADES"

##################################################
# EXPERIMENTAR CON COMBINACIONES DE ATRIBUTOS
#total_rooms SOLO NO DICE MUCHO SI NO SABES CUÁNTAS CASAS HAY -> SE CREAN PROPORCIONES
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

#SE REVISA DE NUEVO LA CORRELACIÓN CON LOS NUEVOS ATRIBUTOS
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
#bedrooms_ratio TIENE MÁS CORRELACIÓN (negativa) QUE total_rooms o total_bedrooms POR SEPARADO
#  -> casas con menor proporción de recámaras/cuartos tienden a ser más caras
