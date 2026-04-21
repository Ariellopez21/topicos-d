# Informe: El Ciclo de un Proyecto de Machine Learning
## Dataset: NSL-KDD — Detección de Intrusiones en Redes

---

### Contexto común (Notebooks 6–10)

Todos los notebooks comparten el mismo bloque de inicialización: rutas `DATA_DIR`, `TRAIN_PATH`, `TEST_PATH`, la lista de 43 columnas `COLUMN_NAMES`, y dos funciones reutilizadas en todos los cuadernos:

- `load_nsl_kdd_txt`: carga los archivos `.txt` del dataset NSL-KDD asignando los nombres de columnas.
- `train_val_test_split`: divide el dataset en train (60%) / val (20%) / test (20%), con soporte para shuffle y stratified sampling.

---

### Notebook 6 — Visualización del Conjunto de Datos

Notebook que presenta herramientas de inspección del dataset.

**Dataset elegido:** NSL-KDD, versión mejorada del KDD Cup 99. Se trata de un dataset ampliamente usado en investigación de IDS (Intrusion Detection Systems). Su elección es adecuada porque tiene un tamaño manejable (125.973 muestras de entrenamiento, 22.544 de test), sin necesidad de muestreo aleatorio, lo que permite comparar resultados de forma consistente.

**Estructura:** 41 atributos de entrada + etiqueta `class` + columna `difficulty`. Los atributos son mixtos: 24 enteros, 15 flotantes, 3 categóricos (`protocol_type`, `service`, `flag`). Ningún valor nulo en el dataset original.

**Distribución de clases observada:** El dataset es desbalanceado. La clase dominante es `normal` (67.343 muestras), seguida de `neptune` (41.214). Existen 22 clases en total, con algunas muy poco representadas (`spy`: 2, `perl`: 3, etc.).

**Análisis de correlaciones:** Para calcularlas, se codifican los atributos categóricos con `LabelEncoder`. Las correlaciones más altas con `class` son negativas para tasas de error SYN (`serror_rate`, `dst_host_serror_rate` ~−0.37) y positivas para `srv_count`, `wrong_fragment`, `dst_host_diff_srv_rate`. El atributo `num_outbound_cmds` resulta constante (NaN en correlación). Se construyen la matriz de correlación completa y una scatter matrix sobre atributos seleccionados.

> Toda la exploración se realiza sobre una copia del training set, sin tocar el test set, evitando que el análisis visual contamine la evaluación futura.

---

### Notebook 7 — División del Conjunto de Datos

Notebook de introducción a los métodos de particionado. Se exploran las variantes de `train_test_split` de sklearn.

**Particionado simple 60/20/20:** Se divide primero en 60% train y 40% restante, luego el 40% se divide en dos partes iguales (val y test). Resultado: 75.583 / 25.195 / 25.195 muestras.

**Problema del shuffle:** Si se usa `shuffle=True` sin semilla fija, cada recarga genera nuevas particiones, con riesgo de que el modelo "vea" todo el dataset iterativamente.

**Stratified Sampling:** Se introduce el parámetro `stratify` sobre `protocol_type` para garantizar que la proporción de cada protocolo (tcp, udp, icmp) se mantiene en los tres subconjuntos. Se verifica gráficamente con histogramas. Finalmente se encapsula en la función `train_val_test_split`.

---

### Notebook 8 — Preparación del Conjunto de Datos

Notebook de preprocesamiento real, con ejecución sobre los datos completos.

**Separación features/etiquetas:** Se separa `X_train` de `y_train` antes de cualquier transformación, principio fundamental para no aplicar transformaciones a las etiquetas.

**Tratamiento de valores nulos (introducidos artificialmente):** Para ilustrar las técnicas, se insertan NaN en `src_bytes` y `dst_bytes` (9.886 filas afectadas). Se presentan tres estrategias:

| Opción | Método | Resultado |
|---|---|---|
| 1 | `dropna()` | Se pierden 9.886 filas |
| 2 | `drop(columns)` | Se eliminan 2 atributos completos |
| 3a | Imputar con **media** | Riesgo: outliers distorsionan la media (`src_bytes` media ≈ 66.914 vs mediana ≈ 43) |
| 3b | Imputar con **mediana** | Más robusto frente a valores extremos |
| 3c | `SimpleImputer(strategy="median")` | Forma canónica sklearn, aplicable solo a columnas numéricas |

> La elección de la mediana sobre la media es relevante dado que `src_bytes` y `dst_bytes` tienen distribuciones muy sesgadas con outliers significativos (max ~1.38 × 10⁹).

**Codificación de atributos categóricos:** Los tres atributos string (`protocol_type`, `service`, `flag`) deben convertirse a numérico antes de cualquier entrenamiento. Se presentan:
- `factorize()` de Pandas y `OrdinalEncoder` de sklearn: asignan enteros secuenciales. Problema: los modelos basados en distancia interpretan falsamente una jerarquía numérica.
- `OneHotEncoder`: genera columnas binarias por categoría (sparse matrix). Parámetro `handle_unknown='ignore'` para manejar categorías no vistas en predicción.
- `pd.get_dummies()`: equivalente más directo para DataFrames.

**Escalado:** Se aplica `RobustScaler` sobre `src_bytes` y `dst_bytes`. Este scaler es robusto frente a outliers (usa IQR en lugar de min/max), lo cual es adecuado dado el rango extremo de estos atributos.

---

### Notebook 9 — Transformadores y Pipelines Personalizados

Construye la infraestructura de preprocesamiento reutilizable sobre la API de sklearn.

#### La API de sklearn (introducción importante del notebook)

El notebook dedica una sección a repasar los tres roles que puede tener un objeto sklearn:

| Rol | Interfaz | Descripción |
|---|---|---|
| **Estimator** | `fit(X)` | Aprende parámetros a partir de datos. Todo objeto sklearn es un estimator. |
| **Transformer** | `fit(X)` + `transform(X)` | Estimator que también transforma datos. `fit_transform(X)` es la forma abreviada. |
| **Predictor** | `fit(X)` + `predict(X)` + `score(X,y)` | Estimator que hace predicciones. |

Un transformador personalizado se crea heredando de `BaseEstimator` + `TransformerMixin`:
- `BaseEstimator` aporta `get_params()` / `set_params()`, necesarios para `GridSearchCV`.
- `TransformerMixin` aporta `fit_transform()` de forma gratuita a partir de `fit()` y `transform()`.

#### Transformadores personalizados para numéricos

**`DeleteNanRows`** — elimina filas con `dropna()`.
```python
class DeleteNanRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):   return self
    def transform(self, X, y=None): return X.dropna()
```
Limitación importante: al eliminar filas, destruye la correspondencia con `y_train`. Solo es seguro usarlo antes de separar etiquetas o cuando val/test no tienen NaN. En esta notebook se usa exclusivamente sobre `X_train` con NaN artificiales, no dentro del `full_pipeline`.

**`CustomScaler`** — aplica `RobustScaler` solo sobre las columnas indicadas, conservando el DataFrame con sus índices originales.
```python
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):  self.attributes = attributes
    def transform(self, X, y=None):
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(X[self.attributes])
        ...
```
> **Problema de implementación:** el `RobustScaler` se instancia y se vuelve a ajustar (`fit_transform`) dentro de `transform()`, no en `fit()`. Esto rompe el contrato sklearn: `fit` debería calcular los parámetros del scaler una sola vez (sobre train), y `transform` debería aplicarlos. Si se usara este transformer en un pipeline sobre val/test, reajustaría el scaler con esos datos, introduciendo **data leakage**. En el notebook se usa de forma manual y secuencial, por lo que no causa problema práctico, pero no es portable a un `Pipeline`.

#### Transformador personalizado para categóricos

**`CustomOneHotEncoding`** — aplica `OneHotEncoder` solo a columnas categóricas y reconstruye el DataFrame completo:
```python
class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._columns = pd.get_dummies(X_cat).columns  # guarda nombres de columnas
        self._oh.fit(X_cat)
        return self
    def transform(self, X, y=None):
        X_num = X.select_dtypes(exclude=['object'])
        X_cat_oh = self._oh.transform(X_cat)           # sparse matrix → DataFrame
        return X_num.join(pd.DataFrame(X_cat_oh, columns=self._columns, index=X.index))
```
Se usa `pd.get_dummies(X_cat).columns` para obtener los nombres de columnas OHE de forma consistente, ya que `OneHotEncoder.get_feature_names_out()` genera nombres con prefijo distinto al de pandas.

> **Nota de compatibilidad:** `OneHotEncoder(sparse=False)` está deprecado en sklearn ≥1.2; la forma correcta es `sparse_output=False`. En NB10 se corrije usando `.toarray()` directamente sobre la salida sparse, que es el patrón más robusto.

#### Pipeline numérico

```
SimpleImputer(strategy="median") → RobustScaler
```

El orden importa: la imputación debe ocurrir antes del escalado porque `RobustScaler` no acepta NaN. Se aplica sobre `X_train_num` (columnas no-object obtenidas con `select_dtypes(exclude=['object'])`).

El resultado de `num_pipeline.fit_transform(X_train_num)` es un `numpy.ndarray`; por eso se reconstruye el DataFrame manualmente:
```python
X_train_prep = pd.DataFrame(X_train_prep, columns=X_train_num.columns, index=X_train_num.index)
```

#### ColumnTransformer (`full_pipeline`)

```python
full_pipeline = ColumnTransformer([
    ("num", num_pipeline,      num_attribs),  # SimpleImputer → RobustScaler
    ("cat", OneHotEncoder(),   cat_attribs),  # OHE sobre protocol_type, service, flag
])
```

El `ColumnTransformer` ejecuta ambas ramas **en paralelo** (sobre copias del mismo DataFrame) y concatena los resultados en un único array. La salida es siempre `numpy.ndarray`, por lo que los nombres de columna se recuperan con `pd.get_dummies(X_train)`. El DataFrame pasa de **42 columnas** originales a **123 columnas** tras la expansión OHE (`service` tiene 70 categorías, `flag` 11, `protocol_type` 3).

> **`Pandas4Warning`:** `select_dtypes(include=['object'])` usada para obtener `cat_attribs` genera un warning en pandas ≥2: en futuras versiones `'object'` no incluirá dtype `'str'` automáticamente. La corrección es `include=['str']`. Afecta a todas las celdas que usen `select_dtypes` con `'object'`.

---

### Notebook 10 — Evaluación de Resultados

Notebook de entrenamiento y evaluación completa del modelo.

#### `DataFramePreparer`: el transformador final

```python
class DataFramePreparer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        num_attribs = list(X.select_dtypes(exclude=['object']))
        cat_attribs = list(X.select_dtypes(include=['object']))
        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline,          num_attribs),
            ("cat", CustomOneHotEncoder(), cat_attribs),
        ])
        self._full_pipeline.fit(X)
        self._columns = pd.get_dummies(X).columns
        return self

    def transform(self, X, y=None):
        X_prep = self._full_pipeline.transform(X.copy())
        return pd.DataFrame(X_prep, columns=self._columns, index=X.index)
```

Diferencias clave con `full_pipeline` del NB9:

| Aspecto | `full_pipeline` (NB9) | `DataFramePreparer` (NB10) |
|---|---|---|
| Tipo | `ColumnTransformer` directo | Transformer que envuelve un `ColumnTransformer` |
| OHE | `OneHotEncoder()` estándar | `CustomOneHotEncoder()` personalizado |
| Salida | `numpy.ndarray` | `pd.DataFrame` (índices y columnas preservados) |
| Portabilidad | No encapsula el descubrimiento de columnas | Auto-detecta num/cat en `fit()` |

**Decisión de diseño crítica — `fit` sobre el dataset completo:**
```python
data_preparer.fit(X_df)           # X_df = dataset completo sin etiquetas
X_train_prep = data_preparer.transform(X_train)
X_val_prep   = data_preparer.transform(X_val)
X_test_prep  = data_preparer.transform(X_test)
```
El `OneHotEncoder` aprende las categorías posibles durante `fit`. Si se ajusta solo con `X_train`, categorías de `service` o `flag` que no aparecen en el subset de entrenamiento generarían columnas ausentes al transformar val/test, rompiendo la dimensionalidad esperada (123 columnas). Ajustar sobre `X_df` garantiza que todos los subconjuntos producen exactamente las mismas 123 columnas.

> Esto es una excepción justificada a la regla de no tocar el test set antes de entrenar: solo se usa para determinar el vocabulario de categorías, no para aprender parámetros estadísticos (media, escala, etc.) que distorsionarían el modelo.

#### Entrenamiento — Regresión Logística

```python
clf = LogisticRegression(solver="newton-cg", max_iter=1000)
clf.fit(X_train_prep, y_train)
```

- **Solver `newton-cg`:** Método de segundo orden (usa la matriz Hessiana). Más costoso por iteración que `lbfgs` o `sag`, pero converge mejor en datasets de alta dimensionalidad (123 features) con distribuciones no triviales. Soporta multiclass (OvR o multinomial).
- **`max_iter=1000`:** El default de sklearn es 100, insuficiente para este dataset. Se aumenta para asegurar convergencia.
- **Clasificación binaria `normal` / `anomaly`:** Las etiquetas `y_train` han sido mapeadas a dos clases antes del entrenamiento (todos los tipos de ataque → `anomaly`). Las métricas usan `pos_label='anomaly'`, indicando que `anomaly` es la clase positiva de interés.

#### Modificaciones realizadas — convergencia y errores de ejecución

Se probaron tres valores de `max_iter` hasta lograr la convergencia:

| Intento | `max_iter` | Resultado |
|---|---|---|
| 1 | 100 (default) | `ConvergenceWarning`: `loss = 0.02391...` — no converge |
| 2 | 500 | `ConvergenceWarning`: `loss = 0.10883...` — no converge |
| 3 | **1000** | Entrenamiento exitoso, sin warnings |

El warning producido en los intentos 1 y 2 fue:
```
ConvergenceWarning: newton-cg failed to converge at loss = <valor>.
Increase the number of iterations.
  warnings.warn(
```

> El solver `newton-cg` necesita más iteraciones en este dataset de 123 dimensiones. Con 100 y 500 iteraciones el algoritmo se detiene antes de alcanzar el criterio de tolerancia (`tol=1e-4` por defecto). Con 1000 iteraciones se alcanza la convergencia.

**Problema pendiente — celdas con `print` fallan tras el entrenamiento:**

Aunque `clf.fit()` con `max_iter=1000` se ejecutó sin errores, todas las celdas posteriores que usan `print` (precisión, recall, F1, evaluación final) generan un error de nombre:

```
NameError: name 'y_pred' is not defined
```

La causa probable es que la variable `y_pred` (resultado de `clf.predict(X_val_prep)`) no fue generada durante la sesión actual del kernel — ya sea porque esa celda no se ejecutó después de los reinicios del kernel durante las pruebas con 100 y 500 iteraciones, o porque el orden de ejecución quedó desincronizado. La solución es ejecutar secuencialmente desde la celda que define `y_pred = clf.predict(X_val_prep)` hacia adelante.

#### Métricas sobre validación (25.195 muestras)

A continución, las métricas, curvas y evaluaciones son realizadas a partir de los resultados registrados en el notebook ejecutado por el profesor antes de ser modificado y obtener errores en los `print`.

| Métrica | Valor | Interpretación |
|---|---|---|
| Precisión | **0.9782** | De cada 100 alertas generadas, ~98 son ataques reales |
| Recall | **0.9649** | De cada 100 ataques reales, ~96 son detectados |
| F1 Score | **0.9715** | Media armónica de precisión y recall |

**Por qué F1 y no accuracy:** El dataset de validación tiene ~11.874 conexiones normales y ~13.321 anómalas (distribución cercana a 50/50 en este subconjunto), por lo que la accuracy no sería engañosa en este caso. Aun así, F1 es más informativo en IDS porque pondera el costo diferenciado de FP y FN.

#### Análisis de la matriz de confusión

```
              Pred: normal   Pred: anomaly
Real: normal     11.457            417      ← Falsos Positivos
Real: anomaly       255          13.066     ← Falsos Negativos
```

| Error | Cantidad | Impacto en IDS |
|---|---|---|
| **Falsos Positivos (FP)** | 417 | Alertas innecesarias → carga operacional |
| **Falsos Negativos (FN)** | 255 | Ataques no detectados → riesgo real de seguridad |

El modelo comete más FP que FN (417 vs 255): es más conservador (prefiere alertar que pasar por alto un ataque). En un IDS esto es el comportamiento deseable, ya que un FN no detectado puede comprometer la red.

#### Curva ROC y Curva PR

**Curva ROC** (`RocCurveDisplay.from_estimator`): Grafica TPR (Recall) vs FPR en todos los umbrales de decisión del clasificador. El área bajo la curva (AUC-ROC) mide la capacidad discriminativa independientemente del umbral. Un valor cercano a 1.0 indica separación casi perfecta entre clases.

**Curva PR** (`PrecisionRecallDisplay.from_estimator`): Grafica Precisión vs Recall al variar el umbral. Más informativa que ROC cuando las clases están desbalanceadas, porque refleja directamente el trade-off entre alertas correctas y cobertura de ataques.

> Ambas curvas se generan con el método `from_estimator`, que internamente llama a `predict_proba` o `decision_function` para obtener los scores continuos necesarios para trazar múltiples puntos de operación.

#### Evaluación final sobre test set

```python
X_test_prep = data_preparer.transform(X_test)
y_pred = clf.predict(X_test_prep)
print("F1 score:", f1_score(y_test, y_pred, pos_label='anomaly'))
# → F1 score: 0.9691633739907233
```

| Subconjunto | F1 Score |
|---|---|
| Validación | 0.9715 |
| Test | **0.9692** |

La caída de 0.0023 puntos entre val y test es estadísticamente insignificante para un dataset de 25.195 muestras. Confirma que:
1. El modelo **generaliza** correctamente (sin overfitting al conjunto de validación).
2. La **estrategia de particionado** 60/20/20 con estratificación produce subconjuntos representativos.
3. El pipeline completo (preprocesamiento + modelo) es **consistente y reproducible** entre subconjuntos.

> **Nota:** Los warnings `Pandas4Warning` aparecen en todas las celdas que llaman a `data_preparer.fit()` o `data_preparer.transform()`, debido al uso de `select_dtypes(include=['object'])` en `DataFramePreparer` y en `CustomOneHotEncoder`. La corrección definitiva es reemplazar `'object'` por `'str'` en ambas clases.

---

### Resumen del ciclo completo

```
Dataset NSL-KDD (125.973 muestras)
    → Exploración y visualización (NB6)
    → División estratificada 60/20/20 (NB7)
    → Limpieza: imputación de NaN con mediana (NB8)
    → Codificación: OneHotEncoder para categóricos (NB8)
    → Escalado: RobustScaler para numéricos (NB8)
    → Encapsulación en Transformadores + Pipeline (NB9)
    → Entrenamiento: LogisticRegression (NB10)
    → Evaluación: F1=0.971 (val) / F1=0.969 (test)
```