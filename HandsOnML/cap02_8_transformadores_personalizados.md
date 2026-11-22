# Cap. 2: Transformadores personalizados

**Libro:** Hands-On Machine Learning (Géron), capítulo 2, dataset de vivienda de California
**Fecha:** 21 de noviembre de 2026
**Script:** `HandsOnML/cap02_8_transformadores_personalizados.py`

> **Idea central:** cuando scikit-learn no trae la transformación que necesitas, puedes crear la tuya. Sigue siendo parte del preprocesamiento de datos.

---

## 1. Log para colas pesadas (`population`)

```python
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])
```

- **`FunctionTransformer`** convierte una función normal en un transformador de scikit-learn, para que después pueda entrar a un `Pipeline`. No necesita `fit` porque el log no aprende nada de los datos.
- **Por qué log:** con cola pesada, unos pocos valores enormes dominan el entrenamiento. El log comprime la cola y la distribución queda más parecida a una campana.
- **Orden:** primero se corrige la forma (log) y después se escala (Standard o min-max). El escalado solo cambia el rango, no la forma.
- Aquí `inverse_func` no se usa, porque `population` es una entrada y no hay nada que regresar.

## 2. `TransformedTargetRegressor`: transformar el target

```python
model = TransformedTargetRegressor(LinearRegression(), transformer=log_transformer)
model.fit(housing[["median_income"]], housing_labels)
predictions_2 = model.predict(some_new_data_2)
```

- Solo transforma el **target** (`housing_labels`, el valor mediano de las casas). `median_income` entra tal cual.
- En `fit` aplica log a los precios y en `predict` aplica `np.exp` para regresar a dólares. Aquí sí se usa `inverse_func`.
- Es la versión automática de lo que se hizo a mano en `cap02_7` con `target_scaler.inverse_transform()`.
- Alternativa sin `log_transformer`: `TransformedTargetRegressor(LinearRegression(), func=np.log, inverse_func=np.exp)`.

> Demo: `some_new_data_2` son 5 distritos del propio train.

## 3. RBF como transformador: columnas de similitud

| Variable | Qué guarda |
|---|---|
| `age_simil_35` | Qué tan cerca está la edad de cada distrito de 35 años (1 = exactamente 35; baja a 0 al alejarse) |
| `sf_simil` | Qué tan cerca está cada distrito de San Francisco, con latitud y longitud juntas (distancia 2D) |

- Ambos son arrays de 16,512 × 1 que se **agregan** como características nuevas; no reemplazan a las originales.
- Los valores son **datos**, no pesos. El modelo aprende en `fit` qué peso darles.
- `kw_args` pasa argumentos extra a `rbf_kernel`: `Y` es el punto de referencia y `gamma`, el ancho de la campana.
- El RBF **no tiene inversa**: 30 y 40 años dan la misma similitud con 35.

## 4. Visto solo en teoría (no está en el script)

- **`ratio_transformer`:** un `FunctionTransformer` que divide una columna entre otra para combinar características.
- **Clases personalizadas:** para transformadores que **aprenden** del train. La plantilla tiene `__init__` (hiperparámetros), `fit` (aprende, guarda atributos con `_` y hace `return self`) y `transform` (aplica). `TransformerMixin` da `fit_transform()` y `BaseEstimator` da `get_params()` / `set_params()`.
- **`StandardScalerClone`:** `StandardScaler` hecho a mano con esa plantilla.
- **`ClusterSimilarity`:** usa KMeans en `fit` para encontrar 10 centros en el mapa y `rbf_kernel` en `transform` para medir la similitud de cada distrito con cada centro (16,512 × 10). Es `sf_simil`, pero con puntos aprendidos. Se usará en el pipeline final del capítulo.

**Vamos en:** pág. 133.
