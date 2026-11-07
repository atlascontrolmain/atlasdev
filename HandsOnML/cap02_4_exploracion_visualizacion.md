# Sesión: Exploración y visualización de datos (Cap. 2, págs. ~100–111)

Archivo de trabajo: `HandsOnML/cap02_4_exploracion_visualizacion.py`

## 1. Repaso de la sesión anterior (muestreo estratificado)

### Por qué `n_splits=10` en `StratifiedShuffleSplit`
- No hay razón matemática: es una convención (y el valor por defecto de Scikit-Learn).
- Compromiso: más divisiones → promedio más estable pero más cómputo; menos divisiones → más rápido pero menos confiable. 5 y 10 son los valores más usados.
- En este capítulo solo se usa el primer par (`[0]`), así que el número no importa todavía.

### Tabla de comparación (Overall vs Stratified vs Random)
- **Overall %:** usa el dataset completo (20,640 distritos).
- **Stratified %:** usa el test set del split estratificado (4,128 distritos).
- **Random %:** usa el test set del split aleatorio (otros 4,128 distritos, distintos).
- No es el mismo 20% medido dos veces: son dos test sets distintos, cada uno comparado contra el total.
- Fórmula por columna: (distritos de la categoría k en ese conjunto) / (total de distritos de ese conjunto). Cada columna da 5 valores (uno por categoría) y suma 100%.
  ```python
  housing["income_cat"].value_counts() / len(housing)
  strat_test_set["income_cat"].value_counts() / len(strat_test_set)
  ```
- No hace falta revisar el train set: `Overall = 0.8 × Train + 0.2 × Test`. Si Test = Overall, entonces Train = Overall automáticamente. Se revisa el test porque es con el que se juzga el modelo al final.

## 2. Explorar solo el conjunto de entrenamiento
- Toda la exploración se hace sobre una copia del train set:
  ```python
  housing = strat_train_set.copy()
  ```
- Razón: evitar el *data snooping bias*. Si exploras el test set, lo que ves influye en tus decisiones y la evaluación final sale más optimista de lo real.
- La copia permite experimentar sin modificar el original.

## 3. Por qué no se puede borrar `load_housing_data()`
- Un `.py` se ejecuta completo de arriba a abajo en cada Run; las variables no persisten entre ejecuciones.
- La copia `housing` se recrea cada vez, así que el script necesita volver a cargar los datos, crear `income_cat`, dividir, quitar `income_cat` y copiar.
- Alternativa futura: mover la función a un módulo aparte (ej. `utils_housing.py`) e importarla.

## 4. Coeficiente de correlación (r de Pearson)
```python
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
```
- `numeric_only=True` es necesario en pandas ≥ 2.0 porque `ocean_proximity` es texto.
- Rango de r: de -1 a 1.
  - **Signo:** positivo = cuando uno crece, el otro también; negativo = cuando uno crece, el otro baja.
  - **Magnitud:** cerca de 1 o -1 = relación lineal fuerte; cerca de 0 = sin relación lineal.
- r solo mide relaciones **lineales**. Una relación curva (ej. en forma de U) puede dar r ≈ 0 aunque exista.
- Resultado clave: `median_income` es el atributo más correlacionado con `median_house_value` (r = 0.69).

## 5. Matriz de dispersión (`scatter_matrix`)
```python
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```
- No calcula r: grafica cada par de atributos para ver la **forma** de la relación.
- Lectura de la cuadrícula: cada celda cruza el atributo de su fila (eje Y) con el de su columna (eje X). Cada punto es un distrito.
  - **Diagonal:** histogramas (un atributo contra sí mismo sería una línea inútil).
  - **Arriba y abajo de la diagonal:** las mismas gráficas con ejes invertidos.
- Cómo deducir r visualmente:
  - Nube alargada que sube → r positivo.
  - Nube alargada que baja → r negativo.
  - Mancha sin dirección → r ≈ 0.
  - Entre más angosta la nube, más cerca de 1 o -1.
- Observaciones en la gráfica:
  - `median_house_value` vs `median_income`: nube claramente inclinada hacia arriba (r = 0.69).
  - vs `total_rooms` y `housing_median_age`: sin dirección (r ≈ 0.14 y 0.10).
  - Línea horizontal en 500,000: valores topados (los distritos más caros se registraron como 500,000).

## 6. Atributos combinados (solo estudiado, no incluido en el código)
- Se crean `rooms_per_house`, `bedrooms_ratio` y `people_per_house`.
- `bedrooms_ratio` (r = -0.26) dice más sobre el precio que `total_bedrooms` o `total_rooms` por separado: menor proporción de recámaras respecto al total de cuartos → vivienda más cara.

## 7. Ajustes al código
- Corregidos los comentarios de `scatter_matrix`: no "obtiene r", visualiza relaciones entre pares de atributos.
- Eliminado el import sin uso de `StratifiedShuffleSplit`.

## Siguiente sesión
- Preparar los datos para los algoritmos de ML (pág. ~111 en adelante): separar predictores y etiquetas (`median_house_value`) partiendo de `strat_train_set`, y limpieza de datos (valores faltantes en `total_bedrooms`).
