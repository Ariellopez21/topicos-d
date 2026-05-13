# Informe Tarea 3: Algoritmos de Machine Learning

---

## Notebook 11 — SVM: Detección de URLs maliciosas

### Objetivo
Clasificar URLs como benignas (`benign`) o de phishing (`phishing`) utilizando
características léxicas extraídas de la URL. Es un problema de clasificación
binaria supervisada con 15.367 muestras y 80 features.

### Por qué se definen funciones auxiliares antes del paso a paso

Se definen dos funciones antes del flujo principal:

**`train_val_test_split`**: encapsula la lógica de doble split (60/20/20)
con soporte de estratificación y semilla reproducible. Se extrae como función
porque: (a) se invoca en un único lugar pero concentra lógica no trivial, y
(b) es idéntica en los notebooks 11, 12 y 13, lo que indica una convención
del profesor para toda la tarea. En Jupyter, además, las funciones deben estar
definidas antes de ser llamadas; agruparlas en una sección propia facilita la
relectura sin interrumpir el flujo del análisis.

**`plot_svc_decision_boundary`**: calcula el hiperplano de decisión de un SVM
lineal a partir del vector de pesos `w` y el intercepto `b`, y dibuja además
los márgenes y los vectores de soporte. Se define antes porque el código
matemático sería ruido visual dentro de la celda de visualización, y porque
se reutiliza en dos secciones distintas del notebook.

### Técnicas y modelos utilizados

| Modelo | Configuración | Dataset | F1 Score |
|---|---|---|---|
| SVM lineal | `C=50` | 2 features | 0.814 |
| SVM lineal + RobustScaler | `C=50` | 2 features | 0.814 |
| SVM lineal | `C=1` | completo (78 feat.) | **0.961** |
| SVM polinomial (Pipeline) | grado 3, `C=20` | 2 features | 0.857 |
| SVM polinomial (`kernel="poly"`) | grado 3, `C=40` | completo | **0.972** |
| SVM RBF | `gamma=0.5, C=1000` | 2 features | 0.862 |
| SVM RBF | `gamma=0.05, C=1000` | completo | **0.964** |

Se utiliza **Pipeline** de scikit-learn para encadenar `RobustScaler` + `SVC`,
lo que garantiza que el escalado se ajusta solo sobre el training set.

### Modificaciones para el entorno local

- El dataset se carga desde `../dataset/Phishing.csv`. Verificar que existe
  `dataset/Phishing.csv` en la raíz del proyecto.
- Se elimina la columna `argPathRatio` manualmente por tener valores infinitos
  antes de usar `SimpleImputer`.
- Los nulos se imputan con la mediana (`SimpleImputer(strategy="median")`).

### Observaciones

- El dataset binario (benign/phishing, casi balanceado) es un subconjunto del
  dataset original de 5 clases (ISCX-URL-2016).
- El kernel polinomial con dataset completo (F1=0.972) es el mejor modelo.
- Para el kernel lineal el escalado no aporta diferencia measurable; para el
  RBF es indispensable ya que la función usa distancias euclidianas.

---

## Notebook 12 — Árboles de Decisión: Detección de Malware en Android

### Objetivo
Clasificar tráfico de red de aplicaciones Android en tres categorías:
`benign`, `adware` y `GeneralMalware`. Clasificación multiclase con un
dataset grande (631.955 registros, 80 features). El objetivo pedagógico
central es demostrar que los árboles **no requieren escalado**.

### Por qué se definen funciones auxiliares antes del paso a paso

Se definen tres funciones:

**`train_val_test_split`**: misma que en el notebook 11. Su repetición
refuerza que los notebooks están diseñados para ejecutarse de forma
independiente, sin importar nada de los demás.

**`remove_labels`**: separa el DataFrame en features `X` y etiquetas `y`.
Se define porque se llama tres veces consecutivas (train, val, test) y
abstraer esas dos líneas (`.drop` + `.copy`) hace el bloque de división
más limpio.

**`evaluate_result`**: recibe predicciones con y sin preprocesamiento y una
métrica, e imprime ambas en una sola llamada. Se define antes porque es la
herramienta de comparación que se usa a lo largo de todo el notebook. Si
estuviera definida más abajo, habría que releer el código hacia atrás para
entender qué imprime.

### Técnicas y modelos utilizados

- **DecisionTreeClassifier** (`max_depth=20`): árbol con profundidad
  máxima para controlar el overfitting parcialmente.
- **RobustScaler**: se incluye únicamente con fines comparativos.
- **plot_decision_boundary** (función local): genera un meshgrid de 1000
  puntos por eje y colorea las regiones predichas. Se usa con solo 2
  features para que sea representable en 2D.
- **graphviz + export_graphviz**: exporta el árbol a formato `.dot` para
  visualizar la estructura de decisión.

| Conjunto | Sin escalar | Con escalado |
|---|---|---|
| Train (F1 weighted) | 0.959 | 0.959 |
| Validación (F1 weighted) | **0.933** | 0.933 |

### Modificaciones para el entorno local

- El dataset se carga desde `'datasets/TotalFeatures-ISCXFlowMeter.csv'`
  (sin `../`). Esto requiere una carpeta `datasets/` dentro de
  `tarea_3_algoritmos/`. Alternativa: cambiar la ruta a
  `'../dataset/TotalFeatures-ISCXFlowMeter.csv'`.
- Requiere `pip install graphviz` (ya incluido como primera celda).

### Observaciones

- El dataset está desbalanceado: 471.597 benign, 155.613 adware, 4.745
  GeneralMalware. La métrica F1 weighted modera el efecto del desbalance.
- La diferencia entre train (0.959) y validación (0.933) con `max_depth=20`
  indica overfitting moderado. Reducir `max_depth` podría mejorar la
  generalización.

---

## Notebook 13 — Random Forests: Detección de Malware en Android

### Objetivo
Extender el notebook 12 comparando un árbol individual con un conjunto de
árboles (Random Forest). Mismo dataset (631.955 registros).

### Por qué se definen funciones auxiliares antes del paso a paso

Funciones idénticas a las del notebook 12: `train_val_test_split`,
`remove_labels`, `evaluate_result`. La razón es la misma: los notebooks
son autosuficientes y la sección "Funciones auxiliares" actúa como un
bloque de utilidades que se ejecuta una sola vez al inicio y queda
disponible para el resto del flujo.

### Técnicas y modelos utilizados (profundizado)

#### DecisionTreeClassifier — árbol individual (baseline)

```python
clf_tree = DecisionTreeClassifier(random_state=42)
```

Sin `max_depth`, el árbol crece hasta pureza máxima en cada hoja. Con
631.955 muestras y 79 features esto produce un árbol profundo que memoriza
el training set (F1 train=0.981 vs F1 val=0.930: diferencia de ~5 puntos,
señal de overfitting).

#### RandomForestClassifier — el modelo central

```python
clf_rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```

**`n_estimators=100`**: número de árboles en el bosque. Cada árbol se
entrena sobre un subconjunto aleatorio de las muestras (muestreo
bootstrap con reemplazo). En clasificación, en cada nodo solo se evalúan
`sqrt(n_features)` ≈ 9 features candidatas (en lugar de las 79 totales).
Esto introduce diversidad entre los árboles, reduciendo su correlación
entre sí.

**`n_jobs=-1`**: utiliza todos los núcleos disponibles para entrenar los
100 árboles en paralelo. Con un dataset de 631k filas y 79 features, sin
este parámetro el entrenamiento puede multiplicar su tiempo por el número
de núcleos.

**Mecanismo de mejora respecto al árbol individual**: el método de
**bagging** (Bootstrap Aggregating) combina las predicciones de los 100
árboles por votación mayoritaria. Los errores de cada árbol son
*independientes* entre sí (porque fueron entrenados en subconjuntos
distintos), por lo que tienden a cancelarse. El resultado es un modelo de
menor varianza que cualquier árbol individual.

| Modelo | F1 Train | F1 Validación |
|---|---|---|
| DecisionTree (sin límite) | 0.981 | 0.930 |
| RandomForest sin escalar | 0.981 | **0.933** |
| RandomForest con escalado | 0.981 | 0.932 |

La mejora de +0.003 en validación puede parecer pequeña, pero el Random
Forest es más robusto ante nuevos datos: su varianza es sistemáticamente
menor que la del árbol individual.

#### Por qué el escalado sigue sin ayudar

Los Random Forests heredan de los árboles de decisión la invarianza al
escalado: en cada nodo se elige un umbral de corte sobre una feature, y
ese umbral es relativo a los valores de esa feature. Multiplicar o sumar
una constante a todos los valores de una feature no cambia el umbral
relativo óptimo.

#### RandomForestRegressor (introducción)

```python
from sklearn.ensemble import RandomForestRegressor
```

El notebook introduce esta variante para problemas de regresión. En lugar
de votación mayoritaria, promedia las salidas numéricas de los árboles.
Aplica el mismo principio de bagging pero con splits evaluados minimizando
el MSE (error cuadrático medio) en lugar de la impureza Gini.

### Modificaciones para el entorno local

- Misma advertencia de ruta del dataset que el notebook 12.
- `n_jobs=-1` es especialmente importante en local para tiempos razonables.

### Observaciones

- La comparación árbol individual vs. Random Forest es el núcleo del
  notebook: muestra empíricamente el beneficio del ensemble con datos reales.

---

## Notebook 17 — KMEANS: Detección de Transacciones Bancarias Fraudulentas

### Objetivo
Aplicar clustering no supervisado (K-Means) para agrupar 284.807
transacciones de tarjeta de crédito e identificar clusters con alta
concentración de fraude. Es el único notebook de **aprendizaje no
supervisado** de la tarea. El dataset contiene 492 fraudes (0.17%),
lo que lo convierte en un problema extremadamente desbalanceado.

### Por qué se definen funciones auxiliares antes del paso a paso

Se definen cuatro funciones específicas para visualización y evaluación:

**`plot_data`**: dibuja un scatter plot con dos colores (negro: legítimas,
rojo: fraudes). Se define antes porque `plot_decision_boundaries` la llama
internamente; si estuviera más abajo, Python lanzaría `NameError`.

**`plot_centroids`**: dibuja los centroides como círculos con una `×`.
Incluye un filtro por `weights` para omitir centroides irrelevantes
(pensado para variantes "soft" de K-Means). Se define antes de cualquier
celda de visualización de clusters.

**`plot_decision_boundaries`**: la función más compleja. Genera un meshgrid
de 1000×1000 puntos, llama a `clusterer.predict()` en cada punto, y rellena
las regiones con `contourf`. Superpone los datos reales y los centroides.
Se define antes porque se usa en la sección 4 y su definición en medio de
las secciones de análisis interrumpiría la lectura.

**`purity_score`**: calcula la pureza de los clusters usando la matriz de
contingencia. Es una métrica **personalizada** no disponible en sklearn.
Se define antes porque es la primera métrica de evaluación que se invoca
en la sección 7, y depende de `sklearn.metrics` que ya fue importado al
principio.

### Técnicas y modelos utilizados (profundizado)

#### KMeans de scikit-learn

```python
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)
```

**Funcionamiento**: inicializa 5 centroides (con K-Means++ por defecto,
que minimiza la probabilidad de mala inicialización), asigna cada punto
al centroide más cercano por distancia euclidiana, recalcula los centroides
como la media de sus puntos, y repite hasta convergencia.

**Elección de K=5**: el notebook fija K=5 sin aplicar el "método del codo"
ni el coeficiente de Silhouette previos. En un proyecto real se evaluarían
varios valores de K y se elegiría el que maximice Silhouette o minimice la
inercia marginal.

**Resultado con 2 features (V10, V14)**:

```
Cluster 1: 5.631 muestras — 429 fraudulentas (7.6% del cluster)
```

Estas dos features tienen correlación ~0.27-0.28 con la clase fraude
(entre las más altas del dataset), lo que explica la buena concentración.

**Resultado con 28 features (sin Time/Amount)**:

Los fraudes se dispersan más. `Time` y `Amount` se eliminan porque están
en escalas completamente diferentes (segundos y dólares) a las features
V1-V28 que ya fueron normalizadas mediante PCA. K-Means usa distancias
euclidianas: una feature con rango 0-172.792 dominaría el cálculo
respecto a features con rango ~[-3, 3].

#### Selección de características con Random Forest (técnica híbrida)

```python
clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd.fit(X, y)
feature_importances = {name: score
    for name, score in zip(list(df), clf_rnd.feature_importances_)}
X_reduced = X[list(feature_importances_sorted.head(7).index)].copy()
```

Se usa un modelo **supervisado** para orientar un proceso **no supervisado**.
La importancia de cada feature (`feature_importances_`) mide la reducción
media de impureza Gini que produce esa feature en todos los árboles. Las 7
más importantes son: V17, V14, V16, V12, V10, V11, V18.

La reducción de 28 a 7 features mejora el clustering porque:
- Elimina features que añaden ruido al cálculo de distancias.
- Reduce la "maldición de la dimensionalidad": en espacios de alta
  dimensión, las distancias euclidianas pierden capacidad discriminativa.

**Resultado con 7 features**:

```
Cluster 3: 308 muestras — 265 fraudulentas (86%)
```

Un cluster altamente puro: el 54% de todos los fraudes se concentra en
solo 308 muestras (0.1% del dataset total).

#### Métricas de evaluación

**Purity Score: 0.999**
```
purity = Σ max_j(n_ij) / N
```
Para cada cluster, suma el conteo de la clase dominante. Como el 99.8%
del dataset son transacciones legítimas, la clase dominante es "legítimo"
en casi todos los clusters. Esta métrica es **engañosa con datasets
desbalanceados**: un clasificador que prediga siempre "legítimo" tendría
Purity Score perfecta. Requiere las etiquetas reales.

**Silhouette Score: 0.181**
```
silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))
```
`a(i)`: distancia media intra-cluster. `b(i)`: distancia media al cluster
más cercano. Rango [-1, 1]; valores cercanos a 0 indican solapamiento entre
clusters. **No requiere etiquetas**. El valor 0.181 es bajo: los clusters
se solapan en el espacio de features, lo que es esperable dado que el fraude
representa solo el 0.17% del dataset.

**Calinski-Harabasz Score: 38.466**
Ratio de dispersión inter-cluster vs. intra-cluster. **No requiere
etiquetas**. No hay un umbral absoluto: se usa para comparar distintas
configuraciones de K.

### Modificaciones para el entorno local

- Dataset cargado desde `../dataset/creditcard.csv`. Verificar que existe.
- Requiere `pip install seaborn` (incluido como primera celda).
- La visualización de histogramas usa `sns.histplot` (ya adaptado desde el
  `distplot` deprecado del código original).

### Observaciones

- Único notebook de aprendizaje no supervisado: el modelo **nunca ve las
  etiquetas** durante el entrenamiento. Las etiquetas solo se usan en la
  evaluación final.
- La técnica híbrida (Random Forest para selección → K-Means con features
  reducidas) es el resultado más interesante del notebook: demuestra que
  combinar un modelo supervisado como auxiliar puede mejorar significativamente
  un modelo no supervisado.
- Con las 7 features correctas, un solo cluster concentra el 54% de todos
  los fraudes con solo 308 muestras, lo que puede usarse como señal de
  alerta en un sistema real.
