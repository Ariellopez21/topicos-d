# Informe — Notebook 5: Regresión Logística para Detección de SPAM

---

## 1. Contexto y problemática

El notebook aborda uno de los problemas clásicos en la intersección del Machine Learning y la ciberseguridad: la detección automática de correos electrónicos de SPAM. El objetivo es construir un clasificador binario capaz de predecir si un correo es spam (1) o ham (0), es decir, legítimo.

El dataset utilizado es el **2007 TREC Public Spam Corpus** (`trec07p`), que contiene **75.419 mensajes** (25.220 ham y 50.199 spam), seleccionado por el profesor para este ejercicio.

El algoritmo empleado es la **Regresión Logística**, que clasifica cada correo como spam o ham. Para poder aplicarlo, el texto de los correos se convierte en vectores numéricos mediante **CountVectorizer**.

---

## 2. Código eliminado: preprocesamiento del corpus TREC crudo

La versión original del notebook incluía un pipeline completo de preprocesamiento diseñado para leer los correos directamente desde los archivos crudos del corpus TREC. Este código fue eliminado en su totalidad, ya que el dataset se obtuvo en formato limpio y estructurado en el archivo `dataset/processed_data.csv`, que contiene el preprocesamiento ya aplicado. A continuación se describe brevemente lo que se eliminó:

- **`MLStripper` y `strip_tags`**: clase y función auxiliar encargadas de limpiar el HTML de los correos, extrayendo únicamente el texto plano.
- **Clase `Parser`**: clase principal de preprocesamiento que, dado un archivo de correo crudo, extraía el asunto y el cuerpo, eliminaba puntuación y stopwords, y aplicaba stemming (reducción de palabras a su raíz) a cada token.
- **`parse_index` y `parse_email`**: funciones para leer el archivo de índice del corpus TREC y parsear individualmente cada correo a partir de su ruta en disco.
- **`create_prep_dataset(index_path, n_elements)`**: función que orquestaba todo el pipeline anterior. Recibía la ruta al índice y el número de correos a procesar, y devolvía los arrays `X` (texto de cada correo) e `y` (etiquetas spam/ham) listos para ser vectorizados. Era el punto de entrada principal al preprocesamiento y la que se invocaba en las secciones de entrenamiento y predicción.

---

## 3. Contenidos del notebook

### Sección 0 — Imports

Instalación de las librerías `scikit-learn` y `pandas` mediante `pip`. Se eliminó la dependencia de `nltk` al no necesitarse ya el preprocesamiento textual.

### Sección 1 — Carga del conjunto de datos

```python
df = pd.read_csv("dataset/processed_data.csv")
```

Carga el CSV completo de 75.419 correos en un DataFrame. Las columnas disponibles son `label` (0=ham, 1=spam), `subject`, `email_to`, `email_from` y `message`. Se imprime el total de correos y la distribución de clases, y se muestra una vista previa con `df.head()`.

### Sección 2 — Preprocesamiento de los datos

El algoritmo de Regresión Logística no puede operar sobre texto directamente, por lo que se utiliza `CountVectorizer` para transformar los correos en vectores numéricos.

**Celda de ejemplo (CountVectorizer sobre un correo):**
```python
sample_text = [str(df['subject'].iloc[0]) + ' ' + str(df['message'].iloc[0])]
vectorizer_sample = CountVectorizer()
vectorizer_sample.fit(sample_text)
```
Se toma el primer correo del dataset, se concatenan su asunto y su cuerpo en una única cadena de texto, y se ajusta un `CountVectorizer` sobre él. El resultado muestra el vocabulario extraído (cada palabra única se convierte en una característica) y el vector de frecuencias correspondiente al correo, donde cada posición indica cuántas veces aparece cada palabra.

### Sección 3 — Entrenamiento del algoritmo

**Preparación del conjunto de entrenamiento:**
```python
df_train = df.head(100)
X_train_raw = (df_train['subject'].fillna('') + ' ' + df_train['message'].fillna('')).tolist()
y_train = df_train['label'].tolist()
```
Se toman los primeros 100 correos del dataset. Para cada uno se concatenan el asunto y el cuerpo (rellenando los valores nulos con cadena vacía) para formar la representación textual. Las etiquetas se extraen directamente de la columna `label`.

**Vectorización:**
```python
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
```
El `CountVectorizer` aprende el vocabulario del conjunto de entrenamiento y transforma cada correo en un vector de frecuencias. Con 100 correos se generan **18.329 características** (palabras únicas). Se muestra la matriz dispersa en forma densa y como DataFrame para visualizar las frecuencias por correo y token.

**Entrenamiento del modelo:**
```python
clf = LogisticRegression()
clf.fit(X_train, y_train)
```
Se instancia el clasificador de Regresión Logística con parámetros por defecto (optimizador lbfgs, regularización L2) y se ajusta sobre la matriz vectorizada y las etiquetas.

### Sección 4 — Predicción

**Preparación del conjunto de test:**
```python
df_150 = df.head(150)
X_all_raw = (df_150['subject'].fillna('') + ' ' + df_150['message'].fillna('')).tolist()
X_test_raw = X_all_raw[100:]
y_test = y_all[100:]
```
Se cargan 150 correos y se reservan los últimos 50 (posiciones 100–149) como conjunto de test, garantizando que no fueron vistos durante el entrenamiento.

**Transformación y predicción:**
```python
X_test = vectorizer.transform(X_test_raw)
y_pred = clf.predict(X_test)
```
Los 50 correos de test se transforman con el mismo `vectorizer` ajustado en la sección anterior (sin refitting, para mantener el mismo espacio de características). El modelo predice la clase de cada correo y se comparan las predicciones con las etiquetas reales.

**Evaluación:**
```python
accuracy_score(y_test, y_pred)
```
Calcula la proporción de predicciones correctas sobre el total.

### Sección 5 — Aumentando el conjunto de datos

Se repite el mismo flujo de las secciones 3 y 4 pero a escala significativamente mayor: se cargan **12.000 correos**, de los cuales 10.000 se usan para entrenamiento y 2.000 para test. El `CountVectorizer` y el `LogisticRegression` se reinstancian para ajustarse al nuevo conjunto sin contaminación del experimento anterior.

---

## 4. Resultados

Con un conjunto de entrenamiento de 100 correos y 50 de test, el modelo alcanzó una **accuracy del 96%**. Al escalar a 10.000 correos de entrenamiento y 2.000 de test, la accuracy subió al **99%**, lo que demuestra que la Regresión Logística se beneficia considerablemente de un mayor volumen de datos. Cabe mencionar que en el experimento a gran escala el optimizador emitió un `ConvergenceWarning` al no alcanzar convergencia completa en 100 iteraciones, aunque esto no afectó la calidad del resultado final.
