# Cap. 2: Transformadores personalizados: `ClusterSimilarity`

**Libro:** Hands-On Machine Learning (Géron), capítulo 2, págs. 133–135, dataset de vivienda de California
**Fecha:** 22 de noviembre de 2026
**Script:** `HandsOnML/cap02_8_transformadores_personalizados3.py`

> **Idea central:** cambiar latitud y longitud crudas por "qué tan cerca estoy de cada zona importante". Lo que más vale la pena de esta sección es aprender el molde de un transformador.

---

## 1. Ideas clave

**Cluster:** un grupo de puntos que están cerca entre sí. KMeans encuentra el centro de cada grupo. Tú le dices cuántos buscar (`n_clusters`), y él decide dónde quedan.

**Qué problema resuelve:** latitud y longitud crudas no le dicen al modelo "este distrito está junto a San Francisco = caro", porque el precio no sube ni baja en línea recta con la latitud. Por eso se cambian por "qué tan cerca estoy de cada zona importante".

**Cómo lo hace (2 pasos):**

1. **`fit`:** KMeans encuentra 10 centros (zonas) en el mapa.
2. **`transform`:** mide la cercanía de cada distrito a cada centro con `rbf_kernel`, y da un valor entre 0 y 1.

**¿Es indispensable?** No. Es un truco para mejorar el modelo (feature engineering). Lo importante de esta sección es aprender el molde de un transformador.

## 2. El molde (esto es lo que vale la pena recordar)

```python
class MiTransformador(BaseEstimator, TransformerMixin):
    def __init__(self, ...):        # solo guarda la configuración (hiperparámetros)
    def fit(self, X, y=None, ...):  # aprende algo de los datos → return self
    def transform(self, X):         # usa lo aprendido para crear columnas nuevas
```

- **`BaseEstimator`** → `get_params()` / `set_params()`, que después usa GridSearchCV.
- **`TransformerMixin`** → te da `fit_transform()` gratis.
- Lo que se aprende en `fit` termina en guion bajo: `kmeans_`.

## 3. Lo que más confundió

### `fit` aprende de las coordenadas, no de los precios

- **`X`** (latitud y longitud) es lo que KMeans agrupa.
- **`sample_weight=housing_labels`** (el precio mediano) solo es la importancia de cada punto: los distritos caros jalan más a los centros hacia ellos.
- Estar cerca de un centro no significa automáticamente "zona cara". Cuál zona es cara lo aprende el modelo después.

### Los datos van en la llamada, nunca en la definición de la clase

```python
# ❌ Mal: mete una variable global en el molde
def fit(self, X, y=None, sample_weight=housing_labels):

# ✅ Bien
def fit(self, X, y=None, sample_weight=None):
...
cluster_simil.fit_transform(housing[["latitude", "longitude"]],
                            sample_weight=housing_labels)
```

Con la versión mala, la clase deja de ser reutilizable y truena en la validación cruzada: entrena con pedazos de los datos, y el tamaño de los pesos ya no coincide.

### `n_clusters` es un hiperparámetro

Nadie "sabe" que son 10: tú lo eliges. Con pocos clusters las zonas quedan enormes; con muchos, hay demasiado detalle. Se elige probando (GridSearchCV, más adelante en el cap. 2; método del codo y silhouette, en el cap. 9).

## 4. Cómo leer la salida

- **`similarities`** es una matriz de 16,512 × 10: una fila por distrito, una columna por cluster.
- **Cada fila** es un vector de 10 valores que describe la ubicación del distrito: qué tan cerca está de cada zona. Reemplaza a latitud y longitud; las demás columnas del distrito siguen aparte.
- **Notación científica:** `9.9e-01` = 0.99 · `4.3e-14` ≈ 0.

Ejemplo (primer distrito, `similarities[:3].round(2)`):

```
[0.08, 0, 0.6, 0, 0, 0.99, 0, 0, 0, 0.14]
```

→ Casi encima del cluster 5 (0.99), medio cerca del 2 (0.6) y lejos de los demás.

> **El número de cada cluster es solo una etiqueta.** Mis valores son idénticos a los del libro (0.99, 0.6, 0.14, 0.08), pero en otras columnas: KMeans encontró los mismos centros y los numeró en otro orden (cambio en sklearn 1.3).

> **Con pesos vs. sin pesos:** cambió poco (0.97 → 0.99, 0.57 → 0.6), porque en California las zonas densas y las caras casi coinciden.

## 5. Graficar los centros (molde mínimo)

```python
centros = cluster_simil.kmeans_.cluster_centers_   # 10 × 2 → [latitud, longitud]

plt.scatter(housing["longitude"], housing["latitude"], s=1)               # distritos
plt.scatter(centros[:, 1], centros[:, 0], marker="X", s=200, c="black")   # centros
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()
```

> ⚠️ Los centros salen en el mismo orden que le pasaste a `fit`. En un mapa, la longitud va en X y la latitud en Y. Si inviertes el orden en un solo lado, las X caen fuera de California.

Graficar no sirve para elegir `n_clusters`; sirve para revisar que el resultado tenga sentido y cachar errores.

---

**Siguiente sesión:** Pipelines de transformación.
