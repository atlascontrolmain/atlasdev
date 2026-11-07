# Capítulo 2 · Parte 3 — División estratificada

Script: [`cap02_3_division_estratificada.py`](cap02_3_division_estratificada.py)
Viene de: [`cap02_2_muestreo_estratificado.md`](cap02_2_muestreo_estratificado.md)
Libro: *Hands-On Machine Learning* (Aurélien Géron), capítulo 2, pág. 99 en adelante.
Punto de pausa: el split estratificado ya está hecho; falta escribir la comparación de proporciones de la Figura 2-10.

---

## Propósito

Apartar un test set que represente bien a la población en la variable que más pesa para predecir el precio: `median_income`. Un split aleatorio puede, por azar, dar un test set no representativo y sesgar la evaluación del modelo.

## Estructura esencial (el flujo que hay que seguir)

```
Cargar datos  →  Crear income_cat  →  Split estratificado  →  (siguiente) Comparar proporciones
                 pd.cut sobre         StratifiedShuffleSplit   Overall vs Stratified vs Random
                 housing completo     o stratify=...
```

La estratificación **no** se hace después del split, sino **en lugar** del split aleatorio. Es el mismo 80/20, pero se toma dentro de cada estrato y luego se juntan las partes:

```
income_cat     total    train (80 %)   test (20 %)
    1            822         657           165
    2          6,581       5,265         1,316
    3          7,236       5,789         1,447
    4          3,639       2,911           728
    5          2,362       1,890           472
             -------      ------         -----
              20,640      16,512         4,128
```

Es **una** estratificación con varios estratos, no varias estratificaciones.

## Cómo correrlo

El script usa rutas relativas, así que se corre desde `atlas_lab`:

```bash
cd /home/breynac/atlas_lab && .envs/bin/python HandsOnML/cap02_3_division_estratificada.py
```

Salida esperada: el `print` de la línea 73 muestra una lista con las dos tablas de `strat_splits[0]`, seguidas. Termina en `[4128 rows x 11 columns]]`. Son 11 columnas porque a las 10 originales se les sumó `income_cat`.

---

## Mapa: tu código ↔ tu resumen

| Líneas | Qué hay en el código | Dónde estudiarlo |
|---|---|---|
| 1-19 | Imports y carga de datos | [Bloque 1](#1-imports-y-carga-líneas-1-23) |
| 25-30 | Split aleatorio, comentado | [Por qué el aleatorio rebarajea](#por-qué-el-split-aleatorio-rebarajea) |
| 32-48 | Split por hash, comentado | [Cómo evita el hash que los datos cambien de conjunto](#cómo-evita-el-hash-que-los-datos-cambien-de-conjunto) |
| 54-55 | `income_cat` con `pd.cut` | [Qué guarda `income_cat`](#qué-guarda-income_cat) |
| 57-60 | Gráfica de los estratos | [Cuántos estratos y de qué ancho](#cuántos-estratos-y-de-qué-ancho) |
| 64-69 | `StratifiedShuffleSplit` y `strat_splits` | [Las 10 divisiones](#stratifiedshufflesplit-y-las-10-divisiones) |
| 72-73 | Elegir la primera división | [Lista vs tupla](#lista-vs-tupla-en-strat_splits) |
| 76-80 | `train_test_split` con `stratify` | [Una división o muchas](#una-división-o-muchas) |
| Aún no escrito | Tabla de comparación | [Figura 2-10](#figura-2-10-comparación-de-sesgo) |

---

## Qué hace cada bloque

### 1. Imports y carga (líneas 1-23)
Igual que en `cap02_1`, con dos cambios en los imports: entra `StratifiedShuffleSplit` y sale `zlib.crc32`. La exploración (`head`, `hist`, `info`) sigue comentada.

### 2. Split aleatorio, comentado (líneas 25-30)
No hace falta para este flujo: el split estratificado lo reemplaza, y `pd.cut` trabaja sobre `housing` completo, no sobre `train_set`/`test_set`. Conviene dejarlo comentado en vez de borrarlo, porque el libro vuelve a hacer un split aleatorio para compararlo contra el estratificado.

### 3. Split por hash, comentado (líneas 32-48)
En el libro, el split final es el estratificado y el hash queda como alternativa explicada. Por eso se conserva como referencia, y el flujo activo queda en carga → `income_cat` → split estratificado.

### 4. Crear `income_cat` (líneas 52-55)
**Este bloque todavía no es el muestreo estratificado.** Solo crea los estratos, pero es obligatorio: para estratificar se necesita una columna categórica, y `median_income` es continua.

- `pd.cut` clasifica cada valor de `median_income` según el intervalo en que cae.
- `bins`: los límites `(0, 1.5]`, `(1.5, 3]`, `(3, 4.5]`, `(4.5, 6]`, `(6, ∞)`.
- `labels`: los nombres 1 a 5 de cada intervalo.
- No se mueve ni se separa ningún dato; solo se agrega una columna.

### 5. Graficar los estratos (líneas 56-60)
Sirve para comprobar que ningún estrato quede demasiado pequeño.

- `value_counts()` cuenta los distritos de cada categoría.
- `sort_index()` los ordena del 1 al 5; si no, saldrían ordenados por frecuencia.
- `plot.bar(rot=0, grid=True)` hace la gráfica. Para verla hay que descomentar `plt.show()`.

### 6. `StratifiedShuffleSplit` (líneas 62-69)
- Línea 64: solo **configura** el divisor; todavía no toca los datos.
  - `n_splits=10`: número de divisiones distintas, cada una un 80/20 estratificado con un barajeo diferente.
  - `test_size=0.2`: el 20 % de cada estrato va a test.
  - `random_state=42`: semilla para que las 10 divisiones sean reproducibles.
- Línea 66: `splitter.split(housing, housing["income_cat"])` hace el trabajo.
  - El primer argumento son los datos, de los que solo usa cuántas filas hay.
  - El segundo es la columna por la que estratifica.
  - Devuelve **índices** (posiciones), no DataFrames, y los entrega de uno en uno en cada vuelta del `for`.
- Líneas 67-68: `housing.iloc[...]` convierte las posiciones en DataFrames. Es `iloc` porque son posiciones, no etiquetas.
- Línea 69: `append` guarda el par `[train, test]`.

Resultado: `strat_splits` es una lista de 10 pares de conjuntos estratificados.

### 7. Elegir la primera división (líneas 71-73)
`strat_splits[0]` es la primera división, `[1]` la segunda, y así hasta `[9]`. Por ahora el libro solo usa la primera, y la línea 72 la desempaca en `strat_train_set` y `strat_test_set`.

### 8. `split_estratificado_comparativo()` (líneas 75-80)
Hace una sola división estratificada con `train_test_split` y `stratify`. La función está definida, pero el `print` de la línea 80 está comentado, así que por ahora no se ejecuta.

---

## Resumen de conceptos

### Por qué estratificar por ingreso
- Por conocimiento del dominio, el ingreso es el mejor predictor del precio de la vivienda. Después se confirma con la correlación.
- El split **cuenta instancias**, no usa medias. La media se conserva sola porque se conservan las proporciones.
- Entre estratos, el reparto está garantizado: el test nunca queda lleno de un estrato y vacío de otro. Las demás variables solo quedan balanceadas por azar.

### Qué guarda `income_cat`
**Duda:** ¿almacena los intervalos con base en `median_income`?
**Corrección:** guarda la **etiqueta** (1 a 5) del intervalo en que cae cada distrito, no los intervalos. Ejemplo: `median_income` = 3.8 → `(3, 4.5]` → `income_cat` = 3.

**Duda:** ¿es una copia de `median_income`?
**Corrección:** no, es una columna **derivada**, y `median_income` queda intacta. Es como sacar una "B" de un 8.7: con `income_cat` no se puede recuperar el valor exacto, solo su rango.

**`np.inf` como último límite.** Es el infinito en punto flotante y funciona como límite "sin techo". En este dataset el máximo es ~15, pero así ningún dato nuevo queda fuera ni sale como `NaN`.

### Cuántos estratos y de qué ancho
- No hay fórmula. Es un balance visto en el histograma: suficientes estratos para capturar la distribución, pero cada uno con suficientes instancias. Alternativa: `pd.qcut`, que corta por cuantiles.
- El sesgo viene de estratos con **pocas** instancias, porque su proporción en el test se vuelve ruidosa. Un estrato grande no es problema; el problema es tener demasiados.
- Para detectarlo se comparan las proporciones de `income_cat` en el total contra las del test set.

### Hash y estratificación resuelven problemas distintos
| Método | Qué evita |
|---|---|
| Hash | Que los datos cambien de conjunto entre versiones del dataset |
| Estratificación | Que el test no represente a la población |

**Duda:** "no lo borro porque hash lo hace aleatorio basado en un umbral".
**Corrección:** el hash **no es aleatorio, es determinista**. Parece aleatorio porque reparte de forma pareja, pero el mismo ID siempre da el mismo hash.

### Cómo evita el hash que los datos cambien de conjunto
Se calcula `crc32` del ID; si el hash es menor que `test_ratio * 2**32`, la instancia va a test. Cada dato decide su conjunto por su ID, sin importar el tamaño del dataset.

Sin hash, nadie "reparte" los datos nuevos: se vuelve a correr `train_test_split` sobre todo el dataset y se reparte desde cero. Eso provoca lo siguiente:

- Datos que estaban en test pasan a train: el modelo aprende de datos con los que antes se evaluó y la métrica sale inflada.
- Datos que estaban en train pasan a test: se evalúa con datos que el modelo ya vio.
- Con cada actualización se repite, y a la larga el modelo ve casi todo el dataset. El test deja de ser "datos nunca vistos".

Con hash, los datos viejos no se mueven y de los nuevos ~20 % cae en test. Ejemplo: 200k + 100k → ~20k de los nuevos van a test.

Con el índice como ID, los datos nuevos solo pueden ir al final y no se deben borrar filas. El libro propone un ID más estable combinando longitud y latitud, aunque puede repetir valores.

El ejemplo visual de la sesión usó un hash de 0 a 99 con umbral 20. El real (`crc32`) va de 0 a 2³², y por eso el umbral es `test_ratio * 2**32`.

### Por qué el split aleatorio rebarajea
`train_test_split` no tiene memoria: cada ejecución reparte desde cero. Por dentro:

```
semilla → secuencia fija de números aleatorios
        → baraja las POSICIONES 0 a n−1
        → el primer 20 % de la baraja va a test
```

El barajeo depende de `n`. Los mismos movimientos de manos sobre 10 cartas o sobre 15 dejan las cartas en lugares totalmente distintos: la semilla fija los movimientos, no dónde queda cada carta. Es decir, los datos no "migran"; es un reparto nuevo que ignora el anterior. El hash no baraja posiciones, sino que le pregunta a cada dato por su ID, y por eso tiene "memoria" sin guardar nada.

### `StratifiedShuffleSplit` y las 10 divisiones
**Duda:** "¿se hacen 10 divisiones, cada una 80/20 y cada 20 se va al 0.2?"
**Corrección:** cada división usa el dataset **completo**. Son 10 repartos distintos del mismo `housing`, y **no** se parte en 10 pedazos. El `0.2` es solo el tamaño del test.

En cada división cambia qué distritos caen en cada conjunto, pero siempre se respetan las proporciones de `income_cat`. Corriendo tu código, la categoría 3 da 35.05 % en el test de las divisiones `[0]`, `[1]` y `[2]`, y 35.06 % en el train de `[0]`.

Se parece a la validación cruzada (evaluar en varias divisiones), con esta diferencia:

| | k-fold | `StratifiedShuffleSplit` |
|---|---|---|
| Cómo reparte | k bloques sin traslape | Sorteos independientes |
| Veces que un dato cae en test | Exactamente una | Varias o ninguna |

A lo segundo se le llama validación cruzada aleatoria (Monte Carlo). En tus 10 divisiones, 2,194 distritos nunca caen en test y alguno cae hasta 8 veces. El libro ve la validación cruzada formal más adelante, aplicada dentro del conjunto de entrenamiento.

### Lista vs tupla en `strat_splits`
- `strat_splits = []` es una **lista** vacía. Una tupla usa paréntesis y no admite `append`, porque es inmutable.
- La lista nunca se convierte en DataFrame: **guarda** DataFrames. Una lista puede contener cualquier objeto.
- `iloc` se aplica a `housing`, no a la lista.
- Al final es una lista de 10 elementos, y cada elemento es un par de DataFrames normales.

### Una división o muchas
| | `train_test_split` + `stratify` | `StratifiedShuffleSplit` |
|---|---|---|
| Da | Una división, en una línea | Las que pidas en `n_splits` |
| Se usa para | Separar el test set final | Ver si el desempeño de un modelo es estable, promediando |

`train_test_split` con `stratify` equivale al `[0]` sin crear las otras 9, porque internamente usa `StratifiedShuffleSplit`. Lo comprobé con tu código y dan exactamente los mismos distritos, en el mismo orden. La utilidad de `StratifiedShuffleSplit` está en recorrer todas las divisiones, no en el `[0]`.

Para comparar, hay que guardar el resultado de la función con otros nombres para no sobrescribir los de la línea 72, y comparar los índices completos, porque el `print` solo muestra 5 filas al inicio y 5 al final:

```python
strat_train_set_x, strat_test_set_y = split_estratificado_comparativo()
print(strat_test_set.index.equals(strat_test_set_y.index))  # True
```

### Figura 2-10: comparación de sesgo
Toda la tabla mide las proporciones de `income_cat`, porque es la variable por la que se estratificó. Estos valores salen de correr el mismo flujo de tu script:

| `income_cat` | Overall % | Stratified % | Random % | Strat. Error % | Rand. Error % |
|---|---|---|---|---|---|
| 1 | 3.98 | 4.00 | 4.24 | 0.36 | 6.45 |
| 2 | 31.88 | 31.88 | 30.74 | -0.02 | -3.59 |
| 3 | 35.06 | 35.05 | 34.52 | -0.01 | -1.53 |
| 4 | 17.63 | 17.64 | 18.41 | 0.03 | 4.42 |
| 5 | 11.44 | 11.43 | 12.09 | -0.08 | 5.63 |

Cómo se obtiene cada columna:

- **Overall:** distritos por categoría en todo `housing`, entre el total (20,640), × 100. Ejemplo: 822 / 20,640 → 3.98 %. Es la referencia, y es el mismo `value_counts()` de la gráfica dividido entre el total.
- **Stratified:** el mismo conteo en `strat_test_set`, entre su tamaño (4,128), × 100.
- **Random:** el mismo conteo en el test set de `train_test_split` sin `stratify`. Este split debe hacerse **después** de crear `income_cat`, porque el `test_set` del inicio del script no tiene esa columna.
- **Error:** desviación relativa respecto a Overall = (porcentaje del test / Overall − 1) × 100. Positivo significa sobrerrepresentada; negativo, subrepresentada. Ojo: el 6.45 sale con los valores sin redondear; con 4.24 / 3.98 daría 6.53.

Cómo leerla:

- El método mejor es el de errores más cercanos a 0.
- Estratificado: errores menores a 0.4 %, prácticamente idéntico al total.
- Aleatorio: errores de hasta 6.45 %, sesgado.
- La categoría 1, el estrato más pequeño, tiene el mayor error en ambos métodos: entre menos instancias, más lo mueve el azar.

La tabla se podría hacer con otra variable categórica, pero la estratificación no garantiza nada para ella. Por ejemplo, en `ocean_proximity` la categoría `INLAND` es 31.74 % del total y 30.28 % en tu `strat_test_set`.

---

## Errores que ya corregiste

**Dos errores que se cancelaban.**
Se había usado `train_size=0.2` en lugar de `test_size=0.2` (hoy línea 64), y en el `append` se metía primero el test y luego el train (hoy línea 69). Como se invertían mutuamente, los tamaños salían "correctos" (16512 y 4128), pero `strat_test_set_n` en realidad tenía el 80 %. Corregir solo uno habría volteado todo. La solución fue usar `test_size` y poner train primero en el `append`.

**`TypeError` por el nombre de la función.**
`TypeError: train_test_split() got an unexpected keyword argument 'test_size'`. Se había definido una función propia llamada `train_test_split`, que reemplaza a la de scikit-learn; dentro de ella, la llamada se llamaba a sí misma, que no acepta argumentos. Por eso hoy se llama `split_estratificado_comparativo` (línea 76).

---

## Detalles por revisar en tu script

No los cambié, para respetar tu código:

1. **Línea 33:** el comentario dice que se quita el hash porque "es aleatorio", pero tu resumen corrige justo eso: es determinista. Se quita porque en este flujo el split final es el estratificado.
2. **Líneas 35-36:** si algún día descomentas el hash, necesitas volver a agregar `from zlib import crc32`.
3. **Sangría:** las líneas 10-17 usan 8 espacios y las 67-69 usan 5. Lo estándar es 4.

---

## Siguiente paso en el script

1. Escribir la comparación de la Figura 2-10: hacer un split aleatorio después de crear `income_cat` y calcular las columnas de la tabla.
2. Quitar `income_cat` de `strat_train_set` y `strat_test_set`, como hace el libro, porque ya no se vuelve a usar.
