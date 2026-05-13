# AUX_TAREA_4 — Patrones reutilizables de tarea_3_algoritmos

Referencia práctica para construir un proyecto propio basado en los
notebooks 11–17. Cada sección incluye código listo para adaptar.

---

## 1. Estructura base del proyecto

Todo notebook de tarea_3 sigue esta secuencia:

```
Imports → Funciones auxiliares → 1.Lectura → 2.Visualización →
3.División → 4.Preparación → 5+.Modelos → Evaluación
```

La sección de funciones auxiliares siempre va **antes** del paso a paso
para que las funciones estén disponibles desde la primera celda de análisis.

---

## 2. Funciones auxiliares reutilizables

### 2.1 Particionado 60/20/20

Presente en los cuatro notebooks. Copiar sin modificar.

```python
from sklearn.model_selection import train_test_split

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate,
        shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate,
        shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)
```

Para datasets desbalanceados, usar `stratify='nombre_columna_etiqueta'`
para mantener proporciones de clases en los tres conjuntos.

### 2.2 Separar features y etiquetas

```python
def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)
```

Uso:
```python
X_train, y_train = remove_labels(train_set, 'nombre_etiqueta')
X_val,   y_val   = remove_labels(val_set,   'nombre_etiqueta')
X_test,  y_test  = remove_labels(test_set,  'nombre_etiqueta')
```

### 2.3 Comparar preprocesamiento vs. sin preprocesamiento

```python
def evaluate_result(y_pred, y, y_prep_pred, y_prep, metric):
    print(metric.__name__, "WITHOUT preparation:",
          metric(y_pred, y, average='weighted'))
    print(metric.__name__, "WITH preparation:",
          metric(y_prep_pred, y_prep, average='weighted'))
```

Uso:
```python
from sklearn.metrics import f1_score
evaluate_result(y_pred, y_val, y_prep_pred, y_val, f1_score)
```

### 2.4 Purity Score para clustering

```python
from sklearn import metrics
import numpy as np

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
```

**Advertencia**: métrica engañosa con datasets muy desbalanceados. Usar
junto con Silhouette o Calinski-Harabasz.

---

## 3. Exploración y visualización del dataset

### 3.1 Inspección inicial

```python
df.head(10)
df.describe()
df.info()
df['columna_etiqueta'].value_counts()   # detectar desbalance
df.isna().any()                          # nulos
df.isin([np.inf, -np.inf]).any()         # infinitos
```

### 3.2 Correlación con la variable objetivo

```python
X = df.copy()
X['etiqueta'] = X['etiqueta'].factorize()[0]  # convertir a numérico
corr_matrix = X.corr()
corr_matrix["etiqueta"].sort_values(ascending=False).head(15)
```

Las features con mayor correlación son candidatas a reducción de
dimensionalidad y a visualización 2D.

### 3.3 Visualización scatter 2D

```python
plt.figure(figsize=(12, 6))
plt.scatter(df["feat1"][df['etiqueta'] == "clase_A"],
            df["feat2"][df['etiqueta'] == "clase_A"], c="r", marker=".")
plt.scatter(df["feat1"][df['etiqueta'] == "clase_B"],
            df["feat2"][df['etiqueta'] == "clase_B"], c="g", marker="x")
plt.xlabel("feat1"); plt.ylabel("feat2")
plt.show()
```

---

## 4. Preparación del dataset

### 4.1 Eliminar columnas con infinitos

```python
# Detectar
is_inf = df.isin([np.inf, -np.inf]).any()
print(is_inf[is_inf])

# Eliminar
X_train = X_train.drop("columna_con_inf", axis=1)
X_val   = X_val.drop("columna_con_inf", axis=1)
X_test  = X_test.drop("columna_con_inf", axis=1)
```

### 4.2 Imputar valores nulos con mediana

```python
from sklearn.impute import SimpleImputer
from pandas import DataFrame

imputer = SimpleImputer(strategy="median")
X_train_prep = imputer.fit_transform(X_train)   # fit solo en train
X_val_prep   = imputer.transform(X_val)
X_test_prep  = imputer.transform(X_test)

# Restaurar como DataFrame
X_train_prep = DataFrame(X_train_prep, columns=X_train.columns,
                          index=X_train.index)
```

### 4.3 Escalado (solo para SVM y KMeans; no para árboles)

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit solo en train
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
```

**`RobustScaler`** usa mediana y IQR: más robusto ante outliers que
`StandardScaler`. Recomendado para datasets de seguridad con distribuciones
muy asimétricas.

---

## 5. Modelos y cuándo usar cada uno

### 5.1 SVM — cuando el dataset es mediano y las clases son separables

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Kernel lineal (rápido, interpretable)
svm_linear = Pipeline([
    ("scaler", RobustScaler()),
    ("svm", SVC(kernel="linear", C=1))
])

# Kernel RBF (mejor para fronteras no lineales)
svm_rbf = Pipeline([
    ("scaler", RobustScaler()),
    ("svm", SVC(kernel="rbf", gamma=0.05, C=1000))
])

svm_linear.fit(X_train, y_train)
y_pred = svm_linear.predict(X_val)
```

Regla práctica de C:
- C pequeño (0.1–1): más margen, más tolerancia a errores → menos overfitting.
- C grande (10–1000): menos margen, intenta clasificar todo → riesgo de overfitting.

**Importante**: SVM requiere escalado obligatorio para kernels RBF y
polinomial. Para el kernel lineal mejora el rendimiento aunque no es
estrictamente necesario.

### 5.2 Árbol de decisión — cuando se necesita interpretabilidad

```python
from sklearn.tree import DecisionTreeClassifier

clf_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
clf_tree.fit(X_train, y_train)
```

- No requiere escalado.
- Ajustar `max_depth` para controlar overfitting (sin límite → overfitting casi seguro en datasets grandes).
- Exportar para visualizar:

```python
from sklearn.tree import export_graphviz
from graphviz import Source

export_graphviz(clf_tree, out_file="arbol.dot",
    feature_names=X_train.columns,
    class_names=clf_tree.classes_,
    rounded=True, filled=True)
Source.from_file("arbol.dot")
```

### 5.3 Random Forest — cuando se prioriza rendimiento sobre interpretabilidad

```python
from sklearn.ensemble import RandomForestClassifier

clf_rnd = RandomForestClassifier(
    n_estimators=100,   # más árboles = menos varianza, más lento
    random_state=42,
    n_jobs=-1           # usar todos los núcleos
)
clf_rnd.fit(X_train, y_train)
```

- No requiere escalado.
- Ventaja clave respecto al árbol individual: menor varianza por bagging.
- Proporciona importancia de features:

```python
feature_importances = pd.Series(
    clf_rnd.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)
print(feature_importances.head(10))
```

### 5.4 K-Means — cuando no hay etiquetas (aprendizaje no supervisado)

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)
```

- **Requiere escalado** (usa distancias euclidianas).
- Eliminar features con escalas muy distintas antes de entrenar.
- Elegir K con el método del codo o Silhouette Score.

```python
# Método del codo para elegir K
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(range(2, 11), inertias, "bo-")
plt.xlabel("K"); plt.ylabel("Inertia")
plt.title("Método del codo")
plt.show()
```

---

## 6. Técnica híbrida: Random Forest para selección de features en KMeans

Patrón del notebook 17: usar un modelo supervisado para reducir dimensionalidad
antes de un modelo no supervisado.

```python
# 1. Entrenar Random Forest supervisado para obtener importancias
clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd.fit(X, y)

# 2. Seleccionar las N features más importantes
feature_importances = pd.Series(
    clf_rnd.feature_importances_, index=X.columns
).sort_values(ascending=False)

N = 7  # ajustar según el problema
X_reduced = X[list(feature_importances.head(N).index)].copy()

# 3. Aplicar KMeans sobre las features reducidas
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_reduced)
```

La reducción mejora KMeans porque elimina el ruido de features poco
informativas y mitiga la maldición de la dimensionalidad.

---

## 7. Evaluación de modelos

### 7.1 Clasificación supervisada

```python
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

# F1 para clasificación binaria
print("F1:", f1_score(y_pred, y_val, pos_label='clase_positiva'))

# F1 para clasificación multiclase (promedio ponderado)
print("F1:", f1_score(y_pred, y_val, average='weighted'))

# Matriz de confusión
print(confusion_matrix(y_val, y_pred))
```

Con datasets desbalanceados, el **F1 Score** es más informativo que accuracy.
Con múltiples clases desbalanceadas, usar `average='weighted'`.

### 7.2 Clustering no supervisado

```python
from sklearn import metrics

# Silhouette (no requiere etiquetas, [-1, 1], mayor = mejor)
sil = metrics.silhouette_score(X_reduced, clusters, sample_size=10000)
print("Silhouette:", sil)

# Calinski-Harabasz (no requiere etiquetas, mayor = mejor)
ch = metrics.calinski_harabasz_score(X_reduced, clusters)
print("Calinski-Harabasz:", ch)

# Purity Score (requiere etiquetas, solo para evaluación post-hoc)
print("Purity:", purity_score(y, clusters))

# Análisis del contenido de cada cluster
from collections import Counter
counter = Counter(clusters.tolist())
bad_counter = Counter(clusters[y == 1].tolist())
for key in sorted(counter.keys()):
    print(f"Cluster {key}: {counter[key]} muestras — {bad_counter[key]} anómalas")
```

---

## 8. Decisiones de diseño para un proyecto nuevo

| Pregunta | Respuesta basada en tarea_3 |
|---|---|
| ¿Cuándo escalar? | Siempre para SVM y KMeans; nunca necesario para árboles/RF |
| ¿Qué scaler usar? | `RobustScaler` para datos con outliers (seguridad/fraude) |
| ¿Cómo manejar nulos? | `SimpleImputer(strategy="median")`, fit solo en train |
| ¿Cómo manejar infinitos? | Eliminar la columna si no es crítica |
| ¿Qué split usar? | 60/20/20 con `train_val_test_split` |
| ¿Dataset desbalanceado? | Usar `stratify` en el split; F1 en lugar de accuracy |
| ¿Overfitting en árbol? | Reducir `max_depth`; pasar a Random Forest |
| ¿KMeans no separa bien? | Reducir dimensionalidad con importancias de RF |
| ¿Cómo elegir K en KMeans? | Método del codo + Silhouette Score |

---

## 9. Datasets disponibles en el proyecto

| Archivo | Notebook de origen | Descripción |
|---|---|---|
| `dataset/Phishing.csv` | 11_SVM | 15.367 URLs, 80 features, binario (benign/phishing) |
| `dataset/creditcard.csv` | 17_KMeans | 284.807 transacciones, 30 features, binario (0/1 fraude) |
| `dataset/TotalFeatures-ISCXFlowMeter.csv` | 12 y 13 | 631.955 flujos de red, 80 features, 3 clases (benign/adware/malware) |

Los tres son datasets de ciberseguridad, por lo que un proyecto propio
puede usar cualquiera de ellos cambiando el modelo o el enfoque de análisis.
