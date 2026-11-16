# Cap. 2: Diseño de scikit-learn y atributos categóricos

**Libro:** Hands-On Machine Learning (Géron), capítulo 2, dataset de vivienda de California
**Fecha:** 15 de noviembre de 2026
**Script:** `HandsOnML/cap02_6_atributos_texto_categoricos.py`

---

## Parte 1: Sesión

### 1. Principios de diseño de scikit-learn

| Principio | Qué significa |
|---|---|
| **Consistencia** | Todos los objetos comparten una interfaz simple y predecible. |
| **Estimadores** | Aprenden de los datos con `fit()`. |
| **Transformadores** | Modifican datos con `transform()` / `fit_transform()`. |
| **Predictores** | Predicen con `predict()` y evalúan con `score()`. |
| **Inspección** | Los hiperparámetros se definen en el constructor y son atributos públicos (`imputer.strategy`). Lo aprendido termina en `_` (`statistics_`, `categories_`). |
| **No proliferación de clases** | Se trabaja con NumPy, SciPy y pandas, no con clases propias. |
| **Composición** | Los objetos se encadenan, por ejemplo en un `Pipeline`. |
| **Valores predeterminados sensatos** | Se puede tener una línea base funcionando rápido. |

> **Duda:** ¿`LinearRegression` es el modelo y `score` su predictor?
> No. `LinearRegression` **es** el predictor (es un tipo de objeto). `predict()` predice y `score()` evalúa (R² en regresión).

### 2. `fit` vs `transform` y fuga de datos

> **Duda:** ¿Con qué medianas se rellena el test set?
> Con las del **entrenamiento**, que quedaron guardadas en `fit`.

> **Duda:** ¿Eso no contamina la evaluación, si son datos que "ya vio"?
> No. La información **entrenamiento → prueba** es legítima, porque así funciona en producción. La fuga de datos es al revés: **prueba → lo que aprende el sistema**. Ejemplo: calcular la mediana con todo el dataset antes de dividirlo.

**Regla:** `fit` solo con entrenamiento; `transform` con todo lo demás (prueba, validación, producción).

### 3. Etapas del proyecto

> **Duda:** ¿Esto ya es entrenar el modelo?
> No, es **preparación de datos / preprocesamiento**. El imputer usa `fit`, pero aprende medianas, no a predecir.

> **Duda:** ¿Mi código ya es un pipeline?
> Es la idea hecha a mano. Un `Pipeline` empaqueta esos pasos en un solo objeto que se ajusta y se aplica como uno solo.

### 4. Atributos categóricos (`ocean_proximity`)

| Categoría | Significado |
|---|---|
| `<1H OCEAN` | A menos de 1 hora del océano |
| `INLAND` | Tierra adentro |
| `ISLAND` | En una isla |
| `NEAR BAY` | Cerca de la bahía |
| `NEAR OCEAN` | Cerca del océano |

- **OrdinalEncoder** (0–4, orden alfabético): solo sirve si las categorías tienen un orden real. Aquí falla porque 0 (`<1H OCEAN`) y 4 (`NEAR OCEAN`) se parecen en la realidad pero quedan numéricamente lejos, y 0 y 1 (`INLAND`) son casi opuestas pero quedan pegadas.
- **OneHotEncoder:** una columna por categoría, con 1 en la de cada distrito y 0 en las demás.

> **Duda:** ¿El one-hot sustituye al ordinal?
> En este problema sí. En general, cada uno sirve para un caso distinto.

**Disperso vs denso:**
- `.toarray()` convierte una salida que ya es dispersa.
- `sparse_output=False` hace que el encoder siempre devuelva arrays densos.
- Con pocas categorías conviene denso; con muchas, dejarlo disperso y usar `.toarray()` solo para ver un pedazo.

**Muchas categorías:** el one-hot genera demasiadas columnas, lo que hace lento el entrenamiento y puede empeorar el rendimiento. Alternativas:
1. Reemplazar la categoría por atributos numéricos con significado (`ocean_proximity` → distancia al océano).
2. Usar otros codificadores del paquete `category_encoders`.
3. Embeddings en redes neuronales (aprendizaje de representaciones, caps. 13 y 17).

### 5. `get_dummies` vs `OneHotEncoder`

> **Duda / confusión:** `df_test` no filtra el dataset real; es un DataFrame **nuevo creado a mano** con 2 filas. **Distritos** = filas; **categorías** = valores posibles.

- `get_dummies` **no tiene `fit`**: crea columnas solo para las categorías que aparecen en los datos que le pasas. Con el train set no hay problema, pero con test o producción (pocas filas, categorías ausentes o nuevas) cambia el número y el orden de las columnas.
- `OneHotEncoder` recuerda las categorías del entrenamiento y siempre da las mismas 5 columnas en el mismo orden. Además se puede meter en un `Pipeline`.
- `handle_unknown="ignore"`: una categoría desconocida sale como fila toda en ceros, en vez de lanzar error.
- `feature_names_in_` = nombre de la columna de entrada.
- `get_feature_names_out()` = nombres de las columnas de salida (`ocean_proximity_<categoría>`).

### 6. Implementación propia

- `OneHotEncoder(sparse_output=False)` aplicado al train set real, con `handle_unknown="ignore"`, y convertido a DataFrame con nombres de columna e índices.
- Resultado: **16,512 × 5**, una fila por distrito.
- Los índices son los IDs originales de cada distrito, desordenados por la división.
- `ISLAND` queda oculta en el `...` de la terminal porque la tabla no cabe a lo ancho.

> Genera una matriz de 16,512 × 5 donde cada fila es el vector one-hot de un distrito, que indica su categoría de `ocean_proximity`.

---

## Parte 2: Código documentado

```python
housing=strat_train_set.drop('median_house_value',axis=1) #X: quitamos lo que se quiere predecir (drop no toca strat_train_set, regresa uno nuevo)
housing_labels=strat_train_set['median_house_value'].copy() #y: lo que se quiere predecir se guarda aparte como etiquetas, copia independiente

# ---- limpieza: rellenar los huecos (NaN) con la mediana ----
imputer = SimpleImputer(strategy="median") # hiperparametro (se elige en el constructor)
housing_num = housing.select_dtypes(include=[np.number]) #elimina ocean_proximity, la mediana solo sale de numeros
imputer.fit(housing_num) #aprende la mediana de cada columna, SOLO con el trainset
imputer.statistics_ #lo que aprendio en fit (por eso termina en _), las medianas guardadas

housing_num.median().values #mismas medianas pero con pandas, para comprobar que coinciden con statistics_
X = imputer.transform(housing_num) #rellena los huecos con las medianas guardadas, regresa array de numpy sin nombres
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index) #lo regreso a dataframe con sus columnas e indices originales

# Transformar los datos de ocean_proximity a valores
housing_cat=housing[['ocean_proximity']] #saca vectores "no traducidos" (doble corchete = dataframe, sklearn lo pide asi)

# ordinal numera 0-4 por orden alfabetico, el modelo creeria que 0 y 1 se parecen mas que 0 y 4 (falso aqui)
ordinal_encoder = OrdinalEncoder() #NO RELEVANTE PARA ESTA RESOLUCIÓN

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)#Nvectores del trainset con ordinal encoder, una sola columna 0-4

# onehot y no get_dummies: el encoder recuerda las 5 categorias del trainset, get_dummies solo ve lo que le pasas
cat_encoder = OneHotEncoder(sparse_output=False)#vectores trainset usando onehot --> lo transforma directo a una matriz (densa, sin .toarray())
housing_cat_1hot = cat_encoder.fit_transform(housing_cat) #fit aprende las categorias, transform crea una columna por categoria con 1 donde cae el distrito
cat_encoder.categories_ #las 5 categorias que aprendio, en el orden de las columnas

cat_encoder.feature_names_in_ #nombre de la columna de entrada: ocean_proximity
cat_encoder.get_feature_names_out() #nombres de las columnas de salida: ocean_proximity_<categoria>
# si llega una categoria que no conoce (nueva o error de dedo) la fila sale en ceros en vez de error
# en el trainset no cambia nada, sirve para test/produccion
cat_encoder.handle_unknown = "ignore"

# matriz de 16,512 x 5 donde cada fila es el vector one-hot de un distrito, que indica su categoria de ocean_proximity
df_output = pd.DataFrame(cat_encoder.transform(housing_cat),
                         columns=cat_encoder.get_feature_names_out(),
                         index=housing_cat.index) #indices originales de cada distrito (desordenados por la division)
print(df_output)
```
