# Capítulo 2 · Parte 2 — Muestreo estratificado

Script: [`cap02_2_muestreo_estratificado.py`](cap02_2_muestreo_estratificado.py)
Viene de: [`cap02_1_carga_y_division_train_test.md`](cap02_1_carga_y_division_train_test.md)

## Qué cambió respecto a cap02_1

La función hecha a mano `shuffle_and_split_data` y `np.random.seed(42)` se reemplazaron por una sola línea de scikit-learn:

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

- Hace lo mismo que la función anterior: revuelve las filas y aparta el 20 % para prueba.
- `random_state=42` reemplaza a `np.random.seed(42)`: fija el revuelto, pero solo para esta llamada.
- Regresa `(train, test)` en ese orden, igual que la función a mano.

Cómo correrlo (desde `atlas_lab`, por las rutas relativas):

```bash
cd /home/breynac/atlas_lab && .envs/bin/python HandsOnML/cap02_2_muestreo_estratificado.py
```

---

## Muestreo estratificado: puntos clave

**La idea.**
Partir `median_income` en rangos (estratos) y hacer el split de forma que el test conserve la misma proporción de cada rango que el dataset completo. Un split puramente aleatorio puede, por azar, dejar algún rango sobrerrepresentado o subrepresentado.

```python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```

`median_income` está en decenas de miles de dólares: `1.5` = 15k, `6.` = 60k.

**Por qué estratificar por ingreso.**
Por conocimiento del dominio: los "expertos" del libro dicen que el ingreso es el mejor predictor del precio de la vivienda. Más adelante se confirma con la correlación: `median_income` es el atributo más correlacionado con `median_house_value`. Regla general: se estratifica por la variable que más pesa en lo que se quiere predecir.

**`np.inf` como último límite.**
`np.inf` es el infinito como número de punto flotante. En `pd.cut` funciona como límite superior "sin techo", así que la categoría 5 es el intervalo (6, ∞): todo ingreso mayor a 60k cae ahí y ningún dato queda fuera. En este dataset `median_income` viene topado en ~15, así que en la práctica la categoría 5 va de 6 a 15. La ventaja de `np.inf` sobre poner `15` es que no hace falta conocer el máximo real: si entra un dato de 20 o de 100, cae en la categoría 5 en lugar de quedar como `NaN` por estar fuera de los límites.

**Cuántos rangos y de qué ancho.**
No hay fórmula; es un balance que se hace mirando el histograma:
- Suficientes rangos para capturar la forma de la distribución.
- Pero cada rango con suficientes instancias.
- Los anchos de 1.5 se eligieron a ojo. Alternativa: `pd.qcut`, que corta por cuantiles y da estratos del mismo tamaño.

**De dónde viene el sesgo: estratos con pocas instancias.**
Un estrato grande no es problema. Si un estrato tiene, por ejemplo, 30 casas, al tomar el 20 % para prueba le tocan unas 6; ahí el azar mueve mucho la proporción y ese grupo queda mal representado. El problema aparece al hacer **demasiados** estratos, porque cada uno se vuelve pequeño. Si un estrato tiene menos de 2 instancias, `train_test_split(..., stratify=...)` falla directamente.

**Cómo verificarlo rápido: comparar proporciones.**

```python
housing["income_cat"].value_counts().sort_index() / len(housing)
test_set["income_cat"].value_counts().sort_index() / len(test_set)
```

Si las proporciones del test se parecen a las del total, el test está bien representado. `housing["income_cat"].value_counts()` sirve además para revisar que ningún estrato sea diminuto. El libro hace esta comparación entre un split aleatorio y uno estratificado: el aleatorio se desvía bastante más en algunas categorías.

---

## Siguiente paso en el script

1. Crear `income_cat` con el `pd.cut` de arriba.
2. Dividir de forma estratificada con `stratify`:

   ```python
   strat_train_set, strat_test_set = train_test_split(
       housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
   ```

3. Comparar las proporciones de `income_cat` del split aleatorio contra el estratificado.
