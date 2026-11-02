# Capítulo 2 · Parte 1 — Carga de datos y división train/test

Script: [`cap02_1_carga_y_division_train_test.py`](cap02_1_carga_y_division_train_test.py)
Libro: *Hands-On Machine Learning* (Aurélien Géron), capítulo 2 — proyecto de precios de vivienda en California.
Punto de pausa: justo antes de **muestreo estratificado** y de pasar a las herramientas de scikit-learn.

---

## Propósito

Hacer "a mano" (con NumPy y pandas, sin scikit-learn) los primeros pasos de un proyecto de ML:

1. Conseguir los datos y cargarlos en un `DataFrame`.
2. Darles un primer vistazo para entender qué hay y qué limitaciones tienen.
3. Apartar un **conjunto de prueba** (test set) antes de hacer cualquier otra cosa, y hacerlo de forma que sea **estable** entre ejecuciones.

La idea es entender qué hacen por dentro las funciones que después da scikit-learn (`train_test_split`, etc.).

## Estructura esencial (el flujo que hay que seguir)

```
Obtener datos  →  Explorar  →  Separar test set  →  (siguiente) Muestreo estratificado
  .tgz → .csv      head/info/hist   aleatorio con semilla        income_cat + scikit-learn
                                    o por hash de un ID
```

El orden importa: el conjunto de prueba se aparta **antes** de explorar a fondo, para que lo que veas de los datos no influya en las decisiones del modelo (ver *data snooping* abajo).

## Cómo correrlo

El script usa rutas relativas (`datasets/housing/...`), así que se corre desde `atlas_lab`, no desde `HandsOnML`:

```bash
cd /home/breynac/atlas_lab && .envs/bin/python HandsOnML/cap02_1_carga_y_division_train_test.py
```

Salida esperada al final: `16512 4128` (filas de entrenamiento y de prueba).

---

## Qué hace cada bloque

### 1. Imports
`pathlib`, `tarfile` y `urllib` para descargar y descomprimir; `pandas` y `numpy` para los datos; `matplotlib` para graficar; `zlib.crc32` para el hash del bloque 5.

### 2. `load_housing_data()` — obtener y cargar los datos
- Si no existe `datasets/housing.tgz`, crea la carpeta `datasets`, descarga el archivo desde el repo de Géron y lo descomprime.
- Siempre termina leyendo `datasets/housing/housing.csv` con `pd.read_csv` y lo regresa como `DataFrame`.
- Así la descarga solo pasa la primera vez; las siguientes ejecuciones leen el archivo local.

### 3. Exploración rápida (comentada)
- `housing.head()` → primeras filas.
- `housing.info()` → número de filas, tipo de cada columna y valores nulos.
- `housing.hist(bins=50)` + `plt.show()` → histograma de cada columna numérica.

### 4. `shuffle_and_split_data(data, test_ratio)` — división aleatoria
- `np.random.permutation(len(data))` revuelve los índices de las filas.
- `test_size = int(len(data) * test_ratio)` → cuántas filas van a prueba (con `0.2`, el 20 %).
- Los primeros `test_size` índices revueltos son de prueba; el resto, de entrenamiento.
- Regresa `(entrenamiento, prueba)` usando `data.iloc[...]` (selección por posición).
- Antes de llamarla se fija `np.random.seed(42)`.
- Los dos `print` dentro de la función son solo de prueba.

### 5. `is_id_in_test_set()` + `split_data_with_id_hash()` — división por hash
- `is_id_in_test_set(identifier, test_ratio)`: calcula `crc32` del ID y decide si cae en prueba comparándolo con `test_ratio * 2**32`.
- `split_data_with_id_hash(data, test_ratio, id_column)`: aplica esa decisión a cada fila (`apply` + `lambda`), obtiene una máscara de `True`/`False` y separa con `data.loc[~máscara]` (entrenamiento) y `data.loc[máscara]` (prueba).

### 6. Crear el ID y dividir
- `housing.reset_index()` agrega una columna `index` con el número de fila, que se usa como ID.
- Se llama a `split_data_with_id_hash(housing_with_id, 0.2, 'index')` y se imprimen los tamaños.

---

## Anotaciones

**Para qué sirve el histograma.**
Muestra límites en cómo se registraron los datos: por ejemplo, `median_house_value` y `housing_median_age` tienen un tope (muchos valores amontonados en el máximo), y `median_income` no está en dólares sino escalado. Detectar eso pronto evita que el modelo aprenda un "techo" artificial.

**Se divide por filas (instancias), nunca por columnas.**
La cantidad de instancias es el número de elementos de cualquier columna (20,640), porque todas las columnas tienen el mismo largo. Por eso `len(data)` cuenta filas, y `test_ratio` se aplica sobre ese número. Cada fila es un caso (un distrito) y las columnas son lo que se sabe de ese caso. El split decide qué casos se usan para aprender y cuáles para evaluar. A un caso de prueba no se le quita información: se aparta completo, con sus 10 columnas.

```
              10 columnas (siempre todas)
           ┌──────────────────────────────┐
train  →   │  16,512 filas  (0.8)         │
           ├──────────────────────────────┤
test   →   │   4,128 filas  (0.2)         │
           └──────────────────────────────┘
```

**Por qué revolver los índices antes de cortar.**
Los datos casi nunca vienen en orden aleatorio, sino en el orden en que se recolectaron (por zona, por fecha…). Este CSV viene ordenado más o menos por geografía: en `housing.head()` las primeras filas son todas `NEAR BAY`, con longitud −122 (área de San Francisco). Si se cortara sin revolver:

```
filas 0 a 16,511      → train   (unas regiones)
filas 16,512 a 20,639 → test    (otras regiones, distintas)
```

El modelo aprendería con unas zonas de California y se evaluaría con otras que nunca vio, así que la evaluación mediría qué tan distintas son las zonas, no qué tan bueno es el modelo. Revolver primero hace que train y test sean, cada uno, una mezcla representativa de todo el dataset.

**Qué son `train_indices` / `test_indices` y cómo los "acomoda" `iloc`.**
Ninguna de las dos listas se "coloca" junto a los datos: son listas aparte de **números de fila**, como una lista de pedidos. La tabla original nunca se mueve. `test_indices` es el 0.2 (20 %) y `train_indices` el 0.8 (80 %). Ejemplo con 10 filas:

```
1) Datos en crudo (data): la tabla no se mueve; la flecha solo indica a dónde irá cada fila

   pos   (10 columnas)
    0    fila 0 completa   → train
    1    fila 1 completa   → train
    2    fila 2 completa   → train
    3    fila 3 completa   → TEST
    4    fila 4 completa   → train
    5    fila 5 completa   → train
    6    fila 6 completa   → train
    7    fila 7 completa   → TEST
    8    fila 8 completa   → train
    9    fila 9 completa   → train

2) El split ocurre en la lista de números, no en la tabla

   shuffled_indices = np.random.permutation(10) = [3, 7, 9, 0, 5, 2, 8, 1, 6, 4]
   test_size        = int(10 * 0.2)             = 2
   test_indices     = shuffled_indices[:2]      = [3, 7]
   train_indices    = shuffled_indices[2:]      = [9, 0, 5, 2, 8, 1, 6, 4]

3) iloc hace la "acomodada": busca cada número en la tabla original
   y copia esa fila completa, en el orden de la lista

   data.iloc[train_indices]  →  train_set: filas 9, 0, 5, 2, 8, 1, 6, 4   (8 filas × 10 columnas)
   data.iloc[test_indices]   →  test_set:  filas 3, 7                     (2 filas × 10 columnas)
```

Las filas de prueba quedan regadas por toda la tabla: eso es lo que logra el revuelto. Y `train_set` empieza con la fila 9, no con la 0, porque `iloc` respeta el orden de la lista. La línea `return data.iloc[train_indices], data.iloc[test_indices]` solo hace el paso 3 dos veces y devuelve las dos tablas nuevas: primero train, luego test.

**Por qué fijar la semilla (`np.random.seed(42)`).**
Sin semilla, cada ejecución genera un test set distinto; tras muchas ejecuciones, el modelo (y tú) terminan viendo todo el dataset. Eso es *data snooping*: tus decisiones se contaminan con datos que deberían ser "nuevos", y el error medido en prueba sale demasiado optimista (parecido al overfitting). Con la semilla, el test set es el mismo en cada ejecución. El `42` no tiene nada especial; solo debe ser siempre el mismo.

**Por qué no basta la semilla y se usa hash.**
La semilla deja de funcionar si el dataset cambia: con más filas, `permutation` produce otro orden y el test set se mezcla con el de entrenamiento. Con el hash, cada fila cae en prueba o entrenamiento según su propio ID, sin importar cuántas filas haya ni cuántas veces se corra el script.

**Cómo funciona el umbral `test_ratio * 2**32`.**
`crc32` regresa un entero entre `0` y `2**32 - 1`, repartido de forma casi uniforme. Si el hash es menor que `0.2 * 2**32`, el ID cae en el 20 % más bajo de ese rango → va a prueba. Por eso el 20 % es **aproximado**: que en este caso dé exactamente `16512 / 4128`, igual que la división aleatoria, es casualidad.

**El orden de los paréntesis en `is_id_in_test_set`.**
Tiene que ser `crc32(np.int64(identifier)) < test_ratio * 2**32`: primero el hash, luego la comparación. Si el `<` queda dentro de `crc32(...)`, se compara primero y se hashea el `True`/`False`. La función regresa enteros en vez de booleanos y, al aplicarles `~`, salen números negativos que pandas interpreta como posiciones de fila inexistentes (`IndexError: positional indexers are out-of-bounds`).

**`loc` vs `iloc`.**
- `iloc` selecciona por **posición** (0, 1, 2, …). Es lo correcto en el bloque 4, donde se tienen arreglos de posiciones revueltas.
- `loc` selecciona por **etiqueta** o con una **máscara booleana**. Es lo correcto en el bloque 5, donde se tiene una serie de `True`/`False`.
- `~` invierte la máscara: `data.loc[~en_prueba]` son las filas que **no** van a prueba.

**Nombres de las variables tras la división aleatoria.**
`shuffle_and_split_data` regresa `(entrenamiento, prueba)`, pero en el script se guardan como `test_indices, train_indices`: el orden está invertido y además son `DataFrame`s, no índices. No afecta el resultado porque esas variables no se usan después, pero conviene tenerlo en cuenta si se reutilizan (un nombre más fiel sería `train_set, test_set`).

**Limitación de usar el número de fila como ID.**
Con `reset_index()`, el ID es la posición de la fila. Esto obliga a que los datos nuevos se agreguen siempre **al final** del dataset y que nunca se borren filas; si no, los IDs se recorren y las filas cambian de conjunto. Si eso no se puede garantizar, hay que construir un ID con columnas estables. El libro propone combinar latitud y longitud: `housing["longitude"] * 1000 + housing["latitude"]`.

**Rutas relativas.**
`Path("datasets/...")` se resuelve desde la carpeta donde **ejecutas** el comando, no desde donde está el script. Por eso se corre desde `atlas_lab`.

---

## Siguiente parte (cap02_2)

**Muestreo estratificado.** El test set debe ser representativo de la población en las variables importantes (en este caso, `median_income`). Para eso se crean categorías de ingreso (`income_cat` con `pd.cut`) y se divide respetando su proporción. Ahí es donde el código pasa a usar scikit-learn (`train_test_split` con `stratify`, `StratifiedShuffleSplit`).
