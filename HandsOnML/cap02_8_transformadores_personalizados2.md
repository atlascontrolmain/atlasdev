# Cap. 2: `StandardScalerClone` (transformador personalizado con clase)

**Libro:** Hands-On Machine Learning (Géron), capítulo 2, dataset de vivienda de California
**Fecha:** 21 de noviembre de 2026
**Script:** `HandsOnML/cap02_8_transformadores_personalizados2.py`

> **Idea central:** el libro reconstruye `StandardScaler` a mano para enseñar la plantilla de un transformador que **aprende**. No es algo que se use en la práctica.

---

## 1. ¿Clon o importar?

Para usar el clon hay que escribir todo el bloque: los imports, la clase completa y después el código con tus datos. Importar `StandardScaler` ahorra todo eso, porque scikit-learn ya escribió la clase, y en una versión más completa (por ejemplo, tolera los NaN).

| | Cuándo |
|---|---|
| `StandardScaler` (importado) | Siempre que necesites estandarizar. Solo escribes el uso: `StandardScaler().fit_transform(...)` |
| `StandardScalerClone` | Nunca en la práctica. Sirve para ver cómo funciona `StandardScaler` por dentro |
| Escribir tu propia clase | Solo cuando scikit-learn **no** trae lo que necesitas (ejemplo: `ClusterSimilarity`). Ahí no hay nada que importar |

## 2. Estructura del script

1. **Imports:** `BaseEstimator`, `TransformerMixin`, `check_array`, `check_is_fitted`.
2. **Definición** (`class StandardScalerClone(...)`): la receta de qué hacen `fit` y `transform`. Por sí sola no toca los datos.
3. **Uso** (`clone = StandardScalerClone()` y lo que sigue): crear el objeto y aplicarlo a `housing_tr`.

## 3. Qué muestra la demo

- **`fit`** aprende 8 medias (`mean_`) y 8 desviaciones (`scale_`), una por columna, cada una por separado.
- **`transform`** aplica `(x − media) / desviación`: todas las columnas quedan con media 0 y desviación 1.
- **`np.allclose(...)` da `True`:** el clon da exactamente lo mismo que `StandardScaler`.
- **Regalos de las clases base:** `get_params()` viene de `BaseEstimator` y `fit_transform()` de `TransformerMixin`.

> Se usa `housing_tr` y no `housing_num`, porque `check_array` rechaza los NaN de `total_bedrooms`.
