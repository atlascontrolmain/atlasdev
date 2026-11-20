# Cap. 2: Escalado y transformación de características

**Libro:** Hands-On Machine Learning (Géron), capítulo 2, dataset de vivienda de California
**Fecha:** 19 de noviembre de 2026
**Script:** `HandsOnML/cap02_7_escalado_transformacion_caracteristicas.py`

> **Idea central:** antes de entrenar, cada columna se revisa y se transforma para que el modelo pueda aprender bien de ella.

---

## Parte 1: Sesión

### 1. Repaso de la sección anterior

- **`SimpleImputer(strategy="median")`**: rellena los valores nulos (por ejemplo, en `total_bedrooms`). Se usa la **mediana** porque los outliers no la jalan como a la media.
- **`OrdinalEncoder` / `OneHotEncoder`**: convierten texto (`ocean_proximity`) en números.
  - `OrdinalEncoder` regresa un array de NumPy denso.
  - `OneHotEncoder` regresa por defecto una **matriz dispersa** de SciPy (`sparse_output=False` para obtenerla densa).

### 2. Escalado y transformación: tres problemas, tres soluciones

| Problema | Ejemplo | Herramienta |
|---|---|---|
| **Escalas muy distintas** | `total_rooms` (6–39,320) vs `median_income` (0–15) | **Min-max** (caja de 0 a 1, frágil con outliers) o **Standard** (centra en 0, es el default) |
| **Cola pesada** (pocos valores enormes) | `population` | **Logaritmo** o raíz cuadrada **antes** de escalar |
| **Varios picos** (distribución multimodal) | `housing_median_age` | **Cubos + one-hot** (cuartos con paredes) o **RBF** (señal de WiFi alrededor de un punto, como 35) |

### 3. Min-max vs Standard

| | `MinMaxScaler` | `StandardScaler` |
|---|---|---|
| Fórmula | `(x − min) / (max − min)` | `(x − media) / desviación estándar` |
| Resultado | Entre 0 y 1 (o el `feature_range` que elijas) | Media 0, desviación 1, sin rango fijo |
| Cuándo usarlo | Cuando el modelo necesita límites fijos (redes neuronales, píxeles) | Opción por defecto (regresión lineal, SVM, KNN, PCA) |
| Debilidad | Muy sensible a outliers | No da rango fijo; rompe matrices dispersas (usar `with_mean=False`) |

### 4. RBF (función de base radial)

Crea una columna nueva que mide **qué tan cerca** está un valor de un punto elegido:

```python
from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
```

- Vale **1** si la edad es exactamente 35 y baja hacia **0** conforme se aleja.
- **`gamma`** controla el ancho de la campana: un gamma grande la hace angosta y uno chico la hace ancha.
- La columna solo mide cercanía. Es el **modelo** quien decide, con el peso que aprende, si estar cerca de 35 sube o baja el precio.

### 5. Escalar el target

También se puede escalar lo que se quiere predecir (`median_house_value`). El modelo predice en la escala transformada y después se regresa a dólares:

```python
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)

scaled_predictions = model.predict(some_new_data)                 # escala estandarizada
predictions = target_scaler.inverse_transform(scaled_predictions)  # dólares
```

Atajo que hace la ida y la vuelta automáticamente:

```python
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
```

> Este bloque es una **demo**: usa una sola columna y 5 distritos del propio train como "datos nuevos".

### 6. Lo que no hay que olvidar

1. Los scalers se ajustan **solo con el train** y después se aplican igual a test y datos nuevos (para evitar *data leakage*).
2. Las transformaciones se deciden **por columna**, viendo su histograma.
3. Esta sección es una **caja de herramientas**. El libro las aplica juntas más adelante con `Pipeline` y `ColumnTransformer`.
