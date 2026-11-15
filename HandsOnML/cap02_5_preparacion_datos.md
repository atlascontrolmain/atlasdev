# Sesión: Preparar los datos para los algoritmos de ML (Cap. 2, pág. ~112 en adelante)

Archivo de trabajo: `HandsOnML/cap02_5_preparacion_datos.py`
Viene de: [`cap02_4_exploracion_visualizacion.md`](cap02_4_exploracion_visualizacion.md)
Alcance de este README: **líneas 40-63** del script.

---

## Por qué escribir funciones en lugar de transformar a mano
El libro abre la sección con cuatro razones. En resumen:
- Repetir las mismas transformaciones en cualquier dataset nuevo sin rehacerlas.
- Ir armando una biblioteca de transformaciones reutilizable.
- Usarlas en el sistema en producción, sobre datos nuevos, antes de pasarlos al modelo.
- Probar varias combinaciones de transformaciones fácilmente.

Esto explica por qué en la línea 60 se usa `SimpleImputer` (un objeto que se entrena y se reaplica) en vez de solo `fillna` a mano.

---

## 1. Revisar r en el train set (líneas 40-43)
```python
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
```
- Es el mismo cálculo de `cap02_4`, aquí solo como punto de partida: `median_income` sigue siendo el atributo más correlacionado (r ≈ 0.69).
- Se calcula **antes** de la línea 45 porque aún necesita la columna `median_house_value`. Después de la línea 45, `housing` ya no la tiene.
- `numeric_only=True` sigue siendo necesario porque `ocean_proximity` es texto.

## 2. Separar predictores y etiquetas (líneas 45-46)
```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```
- **Por qué volver a `strat_train_set`:** el libro pide partir de un train set limpio. La `housing` de la exploración pudo quedar modificada (por ejemplo, con atributos combinados), y no queremos arrastrar esos cambios.
- **Por qué separar:** las transformaciones son para los **predictores** (X), no para la **etiqueta** (y). No tiene sentido imputar, escalar o codificar el valor que queremos predecir con las mismas reglas.
- `drop(..., axis=1)` sin `inplace` **devuelve una copia** sin esa columna; `strat_train_set` queda intacto.
- `.copy()` en las etiquetas evita que `housing_labels` sea una vista de `strat_train_set`: si una se modificara, no afectaría a la otra.
- Resultado: `housing` con 16,512 filas × 9 columnas, `housing_labels` con 16,512 valores. Mismo índice, así cada fila de X sigue alineada con su y.

## 3. Limpieza de datos: tres opciones (líneas 48-58)
La mayoría de los algoritmos de ML no pueden trabajar con valores faltantes. En el train set, `total_bedrooms` tiene **168** celdas vacías (es el único atributo con faltantes).

| Opción | Código | Qué hace | Por qué no se eligió |
|---|---|---|---|
| 1 | `dropna(subset=["total_bedrooms"])` | Elimina los distritos (filas) con el faltante | Pierdes 168 distritos completos con el resto de su información. Además desalinearías `housing` con `housing_labels` si no borras las mismas filas en ambos |
| 2 | `drop("total_bedrooms", axis=1)` | Elimina el atributo (columna) completo | Tiras un atributo entero por 168 celdas de 16,512 |
| 3 | `fillna(median)` | Rellena el faltante con un valor derivado de la columna | **Es la que se usa**: pierde menos información |

Por eso las tres quedan comentadas: son referencia del libro. La opción 3 se implementa en la línea 60 con `SimpleImputer`.

Detalles del código comentado:
- La opción 2 tal como está (`housing.drop(...)` sin asignar ni `inplace`) no modificaría nada: el resultado se descarta.
- En la opción 3, `housing["total_bedrooms"].fillna(median, inplace=True)` da un aviso en pandas ≥ 2.x (asignación encadenada). La forma segura es `housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)`.

### Por qué la mediana y no la media
- La mediana es robusta a valores extremos. `total_bedrooms` tiene una cola larga a la derecha (distritos con muchísimas recámaras), y la media se jalaría hacia esos valores.
- Otras estrategias de `SimpleImputer`: `"mean"`, `"most_frequent"` (sirve también para texto) y `"constant"` (con `fill_value`).

### Por qué la mediana se calcula con el train set
- Se calcula solo con el train set para no espiar el test set (*data snooping*).
- Ese mismo valor hay que guardarlo y reutilizarlo para rellenar el test set y los datos nuevos en producción. Este es el problema que resuelve `SimpleImputer`: guarda el valor por ti.

## 4. `SimpleImputer` (líneas 60-63)
```python
imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
print(imputer.statistics_)
```
- **Línea 60:** crea el imputador. Todavía no ha visto datos; solo sabe que usará la mediana.
- **Línea 61:** `select_dtypes(include=[np.number])` deja solo las columnas numéricas (8 de 9). Se quita `ocean_proximity` porque es texto y no se le puede sacar mediana. El texto se trata más adelante en el libro, aparte.
- **Línea 62:** `fit` calcula la mediana de **cada** columna numérica y la guarda. No modifica `housing_num`.
  - **Por qué todas las columnas y no solo `total_bedrooms`:** hoy solo esa columna tiene faltantes, pero no hay garantía de que los datos nuevos en producción vengan completos. Aplicar el imputador a todo es más seguro.
- **Línea 63:** `statistics_` es donde quedaron guardadas las medianas (el guion bajo final indica, por convención de Scikit-Learn, un atributo aprendido en `fit`).
  ```
  [-118.51  34.26  29.  2125.  434.  1167.  408.  3.5385]
  ```
  Un valor por columna, en el mismo orden que `housing_num.columns`. Por ejemplo, 434 es la mediana de `total_bedrooms`: ese es el valor que ocupará las 168 celdas vacías.

La línea 64 (`housing_num.median().values`) imprime lo mismo calculado a mano. Sirve como comprobación de que el imputador hizo lo que esperábamos.

---

## Patrón que aparece aquí y se repite en todo Scikit-Learn
```
estimador = Clase(hiperparámetros)   →   estimador.fit(datos)   →   estimador.transform(datos)
     (línea 60)                            (línea 62)                (línea 65, fuera de este README)
```
- `fit` **aprende** del train set (aquí, las medianas).
- `transform` **aplica** lo aprendido, a cualquier conjunto: train, test o datos nuevos.
- Separar ambos pasos es lo que permite cumplir las cuatro razones del inicio.
