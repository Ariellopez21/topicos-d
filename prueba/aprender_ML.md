# Guía de Estudio de Machine Learning

> Guía pensada para una prueba **oral y escrita**. Está escrita para que
> entiendas y sepas **explicar con tus palabras**, no solo memorizar.
> Si una parte te parece muy técnica, salta a la frase en **negrita**:
> ahí está la idea que debes saber decir.
>
> Cubre el contenido de los PDF del curso (`cursoML_parte1` y
> `cursoML_parte2`) más los casos prácticos (notebooks 6 al 17).

---

## Índice

1. [Ideas base: ¿qué es Machine Learning?](#1-ideas-base-qué-es-machine-learning)
2. [Vocabulario mínimo (esto te lo van a preguntar)](#2-vocabulario-mínimo)
3. [Tipos de aprendizaje](#3-tipos-de-aprendizaje)
4. [Regresión Lineal](#4-regresión-lineal)
5. [Regresión Logística y Clasificación](#5-regresión-logística-y-clasificación)
6. [Las etapas de un proyecto de ML](#6-las-etapas-de-un-proyecto-de-ml)
7. [Overfitting, Underfitting y Regularización](#7-overfitting-underfitting-y-regularización)
8. [Evaluación de resultados (métricas)](#8-evaluación-de-resultados-métricas)
9. [SVM – Support Vector Machine](#9-svm--support-vector-machine)
10. [Árboles de Decisión](#10-árboles-de-decisión)
11. [Random Forest y Ensemble Learning](#11-random-forest-y-ensemble-learning)
12. [Selección y Extracción de características](#12-selección-y-extracción-de-características)
13. [Clustering (K-Means y DBSCAN)](#13-clustering-k-means-y-dbscan)
14. [Tabla comparativa de algoritmos](#14-tabla-comparativa-de-algoritmos)
15. [Errores comunes que debes EVITAR decir](#15-errores-comunes-que-debes-evitar-decir)
16. [Posibles preguntas orales con respuesta corta](#16-posibles-preguntas-orales)

---

## 1. Ideas base: ¿qué es Machine Learning?

**Definición que puedes decir de memoria:**
> "Machine Learning (aprendizaje automático) es una rama de la Inteligencia
> Artificial donde el computador **aprende a partir de datos**, en lugar de
> seguir reglas escritas a mano por un programador."

La frase clave del curso (Arthur Samuel, 1959): los sistemas aprenden
**"sin ser programados explícitamente"**.

El flujo fundamental es:

```
Experiencia (datos) → Modelo → Predicciones
```

**Jerarquía (te puede caer como diagrama):**

- **Inteligencia Artificial (IA):** lo más amplio, imitar la inteligencia humana.
- **Machine Learning (ML):** un subconjunto de la IA; aprende de datos.
- **Deep Learning:** un subconjunto del ML, inspirado en las neuronas del cerebro.

(IA contiene a ML, y ML contiene a Deep Learning — círculos uno dentro de otro.)

**¿Cuándo conviene usar ML?** (no siempre hace falta)
- Cuando harían falta **demasiadas reglas** para resolver algo a mano.
- En problemas **complejos** donde no se ve una solución clara.
- En entornos que **cambian con frecuencia**.
- Cuando hay **muchísimos datos** difíciles de interpretar por una persona.

---

## 2. Vocabulario mínimo

Esto es lo que más se confunde. Apréndetelo bien porque es la base de todo.

| Término | Qué es | Símbolo |
|---|---|---|
| **Características / features** | Las columnas de entrada del dataset (lo que usas para predecir) | X (X₁, X₂, … Xₙ) |
| **Etiqueta / tag / target** | La columna que quieres predecir (la "respuesta correcta") | Y |
| **Ejemplo / instancia** | Una fila del dataset: una tupla (X, Y) | — |
| **Parámetros** | Los valores que el modelo **aprende** solo durante el entrenamiento | θ (theta) |
| **Hiperparámetros** | Valores que **tú eliges antes** de entrenar (controlan *cómo* aprende) | α, C, K, max_depth… |
| **Función hipótesis (H)** | La fórmula matemática del modelo ya ajustado, la que predice | H(x) o ŷ |
| **Predicción** | La salida del modelo | ŷ ("y gorro") |

> ⚠️ **Aclaración importante (esto en tu cuestionario estaba confuso):**
> La **etiqueta SÍ es una columna del dataset**, simplemente es la columna
> *especial* que queremos predecir. Las demás columnas son las *features*
> (entrada). No es que "esté escondida en otro lado": es una columna más,
> pero es la que hace de "respuesta".

**Frase para la oral:**
> "θ son los parámetros: los **aprende** el modelo. Los hiperparámetros (como
> el learning rate o el número de clusters) los **elijo yo** antes de entrenar.
> Resumen: *los parámetros se aprenden; los hiperparámetros se eligen*."

**La caja negra (modelo mental simple):**

```
Entrada (X = dataset)  →  [ Modelo: parámetros + hiperparámetros ]  →  Salida (ŷ)
```

---

## 3. Tipos de aprendizaje

### Supervisado vs No supervisado (la distinción más importante)

- **Supervisado:** los datos vienen **etiquetados** (sabemos la Y correcta).
  El modelo aprende la relación entrada→salida.
  - **Regresión:** predice un número continuo (ej: el coste de un incidente).
  - **Clasificación:** predice una categoría / valor discreto (ej: spam o no spam).
  - Algoritmos: Regresión Lineal, Regresión Logística, SVM, Árboles, Random Forest.

- **No supervisado:** los datos **NO tienen etiqueta**. El modelo descubre
  **estructura o patrones** por sí solo.
  - Ejemplo: **Clustering** (agrupar) con K-Means o DBSCAN.
  - **Diferencia clave:** se centra en *describir la estructura* de los datos,
    no en *predecir*.

**Frase para la oral:**
> "En supervisado el modelo tiene un 'profesor' que le dice la respuesta
> correcta (la etiqueta). En no supervisado no hay etiquetas: el modelo
> agrupa o encuentra patrones por su cuenta."

### Otras dos clasificaciones que menciona el curso

- **Por lotes (batch) vs en línea (online):**
  - *Batch:* se entrena con **todos los datos a la vez**. Más simple, pero hay
    que reentrenar desde cero para meter datos nuevos.
  - *Online:* se entrena **de forma incremental** (dato a dato o en mini-lotes).
    Se adapta continuamente, pero es más inestable con datos malos.

- **Basado en instancias vs basado en modelos:**
  - *Instancias:* compara los datos nuevos con ejemplos guardados (por similitud).
  - *Modelos:* construye una **función de hipótesis** (una fórmula) para predecir.
    → La Regresión Lineal es un ejemplo de "basado en modelos".

---

## 4. Regresión Lineal

> **Caso práctico asociado:** Notebook 4 — *Predicción del coste de un
> incidente de seguridad*.

### Idea central
**Predice un valor continuo** (un número) trazando la "mejor recta" que
sigue la tendencia de los datos.

> "La regresión lineal hace una **suma ponderada** de las características de
> entrada y le suma una constante (el sesgo). Busca la recta que mejor se
> ajusta a los datos."

### La función hipótesis (te la pueden pedir escrita)

- **Una sola variable (univariable):**
  $$H(x) = θ_0 + θ_1 x$$
  - θ₀ = punto de corte con el eje Y (el **sesgo** o *bias*).
  - θ₁ = pendiente de la recta.

- **Varias variables (multivariable):**
  $$H(x) = θ_0 + θ_1 x_1 + θ_2 x_2 + \dots + θ_n x_n$$

- **Forma vectorial (la más compacta):** $H(x) = θ^T x$

**Cómo recordarlo:** es la ecuación de la recta `y = mx + b` de toda la vida,
pero con muchas variables y con otra notación.

### ¿Cómo "aprende" la recta? (el entrenamiento)

Tres piezas que SIEMPRE aparecen juntas:

1. **Inicializar** los parámetros θ al azar → da una recta cualquiera (mala).
2. **Función de coste (J):** mide **cuán equivocada** está la recta.
3. **Función de optimización:** **ajusta los θ** para reducir ese error,
   repitiendo hasta que el error ya no baja más (**convergencia**).

### Función de coste: Error Cuadrático Medio (ECM / MSE)

$$J(θ) = \frac{1}{2m} \sum_{i=1}^{m} (H_θ(x_i) - y_i)^2$$

**Explicación en palabras (esto es lo que importa):**
1. Para cada ejemplo, resto la predicción menos el valor real → el error.
2. Lo **elevo al cuadrado** (así nunca es negativo y castigo más los errores grandes).
3. Sumo todos los errores.
4. Divido por el número de ejemplos (`m`) para sacar el promedio.
   (Se usa `2m` en vez de `m` por una conveniencia matemática.)

> Cuanto **menor** es J, **mejor** se ajusta el modelo.

### Función de optimización: Descenso de Gradiente

> "El descenso de gradiente ajusta los parámetros paso a paso, en la
> dirección que **reduce el error**, hasta llegar al mínimo."

Regla de actualización:
$$θ = θ - α \cdot \frac{dJ(θ)}{dθ}$$

- **α (learning rate / tasa de aprendizaje):** un **hiperparámetro** que
  controla **qué tan grandes son los pasos**.
  - α muy pequeño → aprende muy lento.
  - α muy grande → puede pasarse, oscilar o nunca converger.
  - α moderado → rápido y estable (lo ideal).

**Imagen mental:** bajar una montaña en niebla. El gradiente te dice hacia
dónde baja la pendiente; α es el tamaño de cada paso que das.

### Variantes (cómo se calcula el error antes de actualizar)

| Método | Calcula el error con… | Actualiza… |
|---|---|---|
| **Batch Gradient Descent** | todos los datos | una vez por época. Preciso pero lento. |
| **Stochastic (SGD)** | un solo ejemplo | tras cada ejemplo. Rápido pero "ruidoso". |
| **Mini-Batch** | grupos pequeños (ej. 32) | tras cada mini-lote. **El más usado** (equilibrio). |

---

## 5. Regresión Logística y Clasificación

> **Caso práctico asociado:** Notebook 5 — *Detección de SPAM*.

### Idea central
A pesar del nombre "regresión", **sirve para CLASIFICAR** (predice valores
discretos: 0 o 1). Ejemplo: ¿es spam (1) o no (0)?

> "La regresión logística toma una combinación lineal de las variables y la
> transforma en una **probabilidad** usando la función sigmoide."

### ¿Por qué no usar regresión lineal para clasificar?
- La recta produce valores continuos que pueden ser **mayores que 1 o menores
  que 0** → no se pueden interpretar como probabilidad.
- Los **valores atípicos (outliers)** distorsionan la recta fácilmente.

### La solución: la función Sigmoide
$$σ(z) = \frac{1}{1 + e^{-z}}$$ donde $$z = θ^T x$$

La sigmoide **transforma cualquier número en un valor entre 0 y 1** (una
probabilidad). Su gráfica tiene **forma de S**.

**Intuición (memoriza estos tres puntos):**
- Si z ≫ 0 → σ(z) ≈ **1** (alta probabilidad de clase 1).
- Si z ≪ 0 → σ(z) ≈ **0** (baja probabilidad).
- Si z = 0 → σ(z) = **0.5** (justo en el límite).

El modelo: $P(y=1 \mid x) = σ(θ^T x)$ → "probabilidad de que sea clase 1".

### Frontera de decisión
- La ecuación $θ^T x = 0$ define la **frontera de decisión**:
  una **recta** (2D), un **plano** (3D) o un **hiperplano** (más dimensiones).
- Un lado de la frontera → clase 1; el otro lado → clase 0.
- **Frase clave:** *"la sigmoide define la probabilidad, pero la recta define
  la decisión"*.

### Función de coste: Log-Loss (no se usa el MSE aquí)
$$J(θ) = -\frac{1}{m}\sum_{i=1}^{m} \left[ y^{(i)} \log(H_θ(x^{(i)})) + (1-y^{(i)}) \log(1 - H_θ(x^{(i)})) \right]$$

**¿Por qué no el MSE como en lineal?** Porque MSE + sigmoide da una función
**no convexa** (con muchos "valles" o mínimos locales donde el algoritmo se
quedaría atascado). El **Log-Loss es convexo**: tiene **un solo mínimo
global**, así que el descenso de gradiente siempre llega al óptimo.

> **Para la oral:** "La regresión logística es un modelo lineal al que se le
> aplica una función no lineal (la sigmoide) para convertirlo en un modelo de
> probabilidades. Y se cambia la función de coste a log-loss para que sea
> convexa y no tenga mínimos locales."

### Comparación rápida Lineal vs Logística

| Elemento | Regresión Lineal | Regresión Logística |
|---|---|---|
| Predice | valores continuos | probabilidad → clase (0/1) |
| Modelo | θᵀx | σ(θᵀx) |
| Función de coste | MSE (Error Cuadrático Medio) | Log-Loss |
| ¿Convexa? | Sí | Sí |

---

## 6. Las etapas de un proyecto de ML

> **Casos prácticos asociados:** Notebooks 6, 7, 8, 9 y 10 (dataset NSL-KDD,
> detección de intrusiones en redes). Analizados en
> `informe_ciclo_completo_proyecto_ML.md`.

**Idea importante de partida:** en proyectos reales, la mayor parte del tiempo
se va en **preparar y transformar los datos**, no en el algoritmo. *La calidad
y cantidad de datos importan tanto como el propio algoritmo.*

### Recomendaciones para un buen dataset
- Tener **suficientes datos** (con muchos datos, hasta importa menos qué algoritmo uses).
- Que sean **representativos** del problema real.
- Que sean **datos reales y de calidad** (no inventados a mano).
- Usar **características relevantes** y minimizar las irrelevantes.

### Etapa A — Visualización del conjunto de datos (Notebook 6)
Inspeccionar y entender los datos **antes** de tocarlos:
- Ver las primeras filas, `info()` (tipos), `describe()` (estadísticas).
- `value_counts()` para categóricas (ej: cuántos "normal" vs "anómalo").
- Histogramas para ver distribuciones.
- **Análisis de correlación:** ver qué features se relacionan con la etiqueta
  y entre sí. Una matriz de correlación; si dos features están muy
  correlacionadas, una puede sobrar.
- *Lectura de formatos:* `.txt`/CSV con `read_csv` de Pandas; `.ARFF` necesita
  una librería extra (`liac-arff`).

### Etapa B — División del conjunto de datos (Notebook 7)
Se divide en **3 subconjuntos**:

```
Data Set
 ├── Train Set (60%–90%)  → para ENTRENAR (aprender los parámetros)
 │     └── (a su vez se subdivide en train + validación)
 ├── Validation Set       → para ELEGIR el mejor modelo / hiperparámetros
 └── Test Set (10%–40%)   → para la EVALUACIÓN FINAL (datos nunca vistos)
```

Resumen para memorizar:
- **Train → aprender.**
- **Validation → elegir el mejor modelo.**
- **Test → evaluar de forma final.**

> ⚠️ **Regla de oro:** **NUNCA** se debe mirar ni modificar el **Test Set**
> durante el entrenamiento. Es la prueba "limpia" del mundo real.

Dos parámetros del split:
- **Shuffle:** mezcla los datos para evitar sesgos por el orden.
- **Stratify (estratificación):** asegura que cada subconjunto mantenga la
  **misma proporción de clases** que el original.
- (Ojo: shuffle y stratify son *parámetros del proceso*, no hiperparámetros
  del modelo.)

### Etapa C — Preparación / Limpieza de datos (Notebook 8)
Los algoritmos **no toleran valores nulos** y casi siempre **necesitan
números**. Por eso:

1. **Valores faltantes (nulos):** tres opciones —
   - Eliminar la fila (el ejemplo).
   - Eliminar la columna (la característica).
   - **Imputar** un valor (la media o, mejor, la **mediana** si hay outliers).

2. **Datos categóricos → numéricos:**
   - **One-Hot Encoding:** crea un bit por categoría y activa uno.
     Ej: PayPal = `[1 0 0]`, Tarjeta = `[0 1 0]`, Bizum = `[0 0 1]`.
   - **Dummy Coding:** parecido pero con K-1 bits (ahorra una columna).

3. **Escalado** (para features con rangos muy distintos):
   - **Min-Max (Normalización):** lleva los valores al rango [0, 1].
   - **Estandarización:** deja media 0 y desviación estándar 1.
   - (En el caso práctico se usó `RobustScaler`, bueno frente a outliers.)

4. **Desequilibrio de clases (data imbalance):** cuando una clase es mucho
   más frecuente (ej: fraude vs legítimo):
   - **Over-sampling:** duplicar la clase minoritaria.
   - **Under-sampling:** reducir la clase mayoritaria.
   - Modificar la función de error para dar más peso a la minoritaria.

> ⚠️ **Regla crítica de las transformaciones:** se ajustan (se "aprenden")
> **solo con el Train Set**, y luego se aplican igual a validación y test.
> Si las ajustas con todos los datos, cometes **fuga de datos (data leakage)**.

### Etapa D — Transformadores y Pipelines (Notebook 9)
Hacer las transformaciones una por una es **tedioso y propenso a errores**.
Scikit-learn ofrece:

- **La API de sklearn tiene 3 roles:**
  - *Estimador:* aprende parámetros con `fit()`.
  - *Transformador:* además transforma datos con `transform()`.
  - *Predictor:* hace predicciones con `predict()`.
- **Pipeline:** encadena varios pasos; la salida de uno es la entrada del siguiente.
- **ColumnTransformer:** aplica pipelines **distintos en paralelo** a columnas
  distintas (ej: escalar las numéricas y One-Hot las categóricas) y luego
  une los resultados.
- Puedes crear **transformadores propios** heredando de `BaseEstimator` y
  `TransformerMixin`, implementando `fit()` y `transform()`.

### Etapa E — Entrenamiento y Evaluación (Notebook 10)
Se entrena el modelo (en el caso práctico, una **Regresión Logística** para
clasificar tráfico como `normal` o `anomaly`) y se evalúa con métricas
(ver sección 8). Resultado del caso: F1 ≈ 0.97 en validación y test → el
modelo **generaliza bien**.

---

## 7. Overfitting, Underfitting y Regularización

Este tema es **clásico de prueba oral**. Domínalo.

### Overfitting (sobreajuste)
> "El modelo **memoriza** los datos de entrenamiento en lugar de aprender los
> patrones. Va muy bien en entrenamiento pero **falla con datos nuevos**."

- Causa típica: modelo demasiado **flexible** (muchas features, árbol muy profundo…).
- Es **memorización**, no aprendizaje.

### Underfitting (subajuste)
> "El modelo es **demasiado simple/rígido** para captar el patrón. Falla
> incluso en los datos de entrenamiento." (Ej: una recta para datos curvos.)

```
Underfitting  →  Buen ajuste  →  Overfitting
(muy simple)     (correcto)      (muy complejo)
```

### ¿Cómo se DETECTA el overfitting? (¡pregunta muy frecuente!)
Comparando el rendimiento en train vs validación:

> **Si el rendimiento en ENTRENAMIENTO es MUCHO MEJOR que en VALIDACIÓN,
> hay overfitting.**
>
> En términos de error: error de train muy bajo PERO error de validación alto.
> En términos de exactitud: **acc_train ≫ acc_val** (entrenamiento mucho
> mayor que validación).

> 🔴 **CORRECCIÓN IMPORTANTE:** En tu `cuestionario.md` esto quedó escrito al
> revés (`acc_train << acc_val`). Lo correcto es **`acc_train >> acc_val`**:
> el entrenamiento da MUCHO MÁS que validación. Tiene lógica: el modelo se sabe
> "de memoria" los datos de entrenamiento (acierta casi todo ahí) pero falla en
> los de validación que no ha visto. **No te equivoques en esto en la prueba.**

(En el Notebook 13 se vio en la práctica: F1 train = 0.981 vs F1 validación =
0.930 → señal de overfitting.)

### Soluciones al Overfitting
1. **Más datos** de entrenamiento.
2. **Reducir el número de características** (selección o extracción de features).
3. **Regularización** (la más común).

### Regularización
> "Añade una **penalización** a la función de coste para que los parámetros
> (θ) **no se hagan demasiado grandes**, reduciendo la flexibilidad del modelo."

- La controla un hiperparámetro **λ (lambda)**:
  - λ = 0 → sin regularización (riesgo de overfitting).
  - λ muy grande → modelo demasiado simple (riesgo de underfitting).
- Idea: `Error total = Error de entrenamiento + λ · Complejidad del modelo`.

**Dos tipos:**
- **L2 (Ridge):** penaliza la suma de los **cuadrados** de los θ. Hace el
  modelo más "suave" y **menos sensible al ruido** (outliers, errores de
  medición). *Pesos grandes = modelo "nervioso"; pesos pequeños = modelo estable.*
- **L1 (Lasso):** penaliza la suma de los **valores absolutos** de los θ.
  Puede llevar algunos θ exactamente a 0 (sirve para seleccionar features).

---

## 8. Evaluación de resultados (métricas)

> **Caso práctico:** Notebook 10.

Cuando hay muchísimos datos no puedes revisar a mano: usas **métricas** que
comparan las predicciones con las etiquetas reales.

### La Matriz de Confusión (la base de todo)
Para clasificación. Compara lo que el modelo predijo vs la verdad:

|  | Predijo Positivo | Predijo Negativo |
|---|---|---|
| **Real Positivo** | VP (acierto) | FN (se le escapó) |
| **Real Negativo** | FP (falsa alarma) | VN (acierto) |

- **VP (Verdadero Positivo):** dijo "fraude" y era fraude. ✅
- **VN (Verdadero Negativo):** dijo "legítimo" y era legítimo. ✅
- **FP (Falso Positivo):** dijo "fraude" pero era legítimo. ❌ (falsa alarma)
- **FN (Falso Negativo):** dijo "legítimo" pero era fraude. ❌ (¡se le escapó!)

### Las métricas principales

- **Precisión (Precision):** de todo lo que predije como positivo, ¿cuánto
  acerté? → `VP / (VP + FP)`. *"De cada 100 alarmas, ¿cuántas eran reales?"*
- **Exhaustividad / Recall:** de todos los positivos reales, ¿cuántos detecté?
  → `VP / (VP + FN)`. *"De cada 100 ataques reales, ¿cuántos pillé?"*
- **F1-Score:** combina precisión y recall en un solo número (su media
  armónica). Va de 0 (peor) a 1 (mejor). Un F1 de 0.9 ≈ clasifica bien el 90%.
  - **¿Por qué F1 y no solo accuracy?** Porque con **clases desbalanceadas**
    la accuracy engaña (si el 99% es "legítimo", un modelo que siempre diga
    "legítimo" tendría 99% accuracy pero sería inútil). El F1 weighted modera
    el desbalance.

### Métricas gráficas
- **Curva ROC:** grafica la **tasa de Verdaderos Positivos (eje Y)** vs la
  **tasa de Falsos Positivos (eje X)** al variar el umbral. El punto ideal es
  la **esquina superior izquierda**. Cuanta **más área bajo la curva (AUC)**,
  mejor el modelo.
- **Curva PR (Precisión-Recall):** más informativa que la ROC cuando las
  clases están **desbalanceadas**.

---

## 9. SVM – Support Vector Machine

> **Caso práctico:** Notebook 11 — *Detección de URLs maliciosas*
> (benignas vs phishing). Analizado en `informe_tarea_3.md`.

### Idea central
> "SVM busca el **hiperplano que separa las clases dejando el mayor margen
> posible** entre los puntos más cercanos de cada clase. A más margen, mejor
> generaliza."

- Los puntos más cercanos al límite que "sostienen" el margen son los
  **vectores de soporte (support vectors)**.
- A esto se le llama **"clasificación de amplio margen"** (*large margin*).
- Destaca con datasets **pequeños y complejos**. Hace regresión y clasificación.

### Hard Margin vs Soft Margin
- **Hard Margin:** exige separar las clases **sin ningún error**. Solo sirve
  si los datos son **perfectamente separables** y es **muy sensible a outliers**.
- **Soft Margin:** permite **algunos errores** a cambio de un mejor margen
  general. Tolera outliers y ruido. **Es lo que se usa en la práctica.**
- **Importante:** no son dos algoritmos distintos. Es el **mismo SVM** regulado
  por el hiperparámetro **C** (sklearn siempre usa soft margin).

### Hiperparámetros clave (¡muy preguntados!)
- **C:** controla el equilibrio entre **margen amplio** y **pocos errores**.
  - C bajo → más errores permitidos, margen amplio → modelo **más general** (más regularización).
  - C alto → exige clasificar casi todo bien → riesgo de **overfitting** (casi hard margin).
- **kernel:** la forma de la frontera (lineal, polinómico, RBF…).
- **gamma** (en kernel RBF): el **radio de influencia** de cada punto.
  - gamma alto → frontera muy irregular (overfitting).
  - gamma bajo → frontera más suave.

### Kernels (para datos NO separables linealmente)
Cuando no puedes separar con una recta, el **"kernel trick"** proyecta los
datos a una **dimensión superior** donde sí se pueden separar.

- **Kernel lineal:** frontera recta. Para datos separables linealmente.
- **Kernel polinómico:** añade features como x², x³ → fronteras curvas.
- **Kernel RBF (Gaussiano):** usa **puntos de referencia (landmarks)** y mide
  la **similitud** (distancia) de cada ejemplo a ellos.
  - Si un punto está **cerca** de un landmark → nueva feature ≈ 1.
  - Si está **lejos** → nueva feature ≈ 0.
  - Permite fronteras no lineales complejas.

> ⚠️ **El escalado es obligatorio para el kernel RBF** (usa distancias
> euclidianas). Para el kernel lineal no cambia el resultado.

---

## 10. Árboles de Decisión

> **Caso práctico:** Notebook 12 — *Detección de malware en Android*
> (benign / adware / GeneralMalware). Clasificación multiclase.

### Idea central
> "Un árbol de decisión clasifica haciendo una **serie de preguntas
> binarias** (sí/no) sobre las características, dividiendo los datos hasta
> llegar a una predicción."

Ejemplo de regla: *"¿X1 ≤ 3?"* → si no, *"¿X2 ≤ 4?"* → … → clase final.

- Cada **nodo** = una pregunta/corte sobre una feature.
- Cada **hoja** = una clase final.
- Crean **fronteras de decisión no lineales** (a base de segmentos), así que
  sirven para datos **no separables linealmente** (al contrario que la
  regresión logística o el SVM lineal, que hacen líneas/planos rectos).
- Sirven para **clasificación y regresión**.
- **Son los más interpretables:** cada rama es una regla legible.

### ¿Cómo elige las preguntas? La Impureza de Gini
**Gini** mide **cuán "mezcladas"** están las clases en un nodo:
- Gini **bajo (≈ 0)** → el nodo es casi todo de una sola clase → "ordenado/puro".
- Gini **alto (≈ 0.5 en binario)** → mezcla de clases → "desordenado".

Fórmula: $Gini = 1 - \sum_{i=1}^{K} p_i^2$ (pᵢ = proporción de la clase i).

Ejemplo: nodo con 80% clase A y 20% clase B →
`Gini = 1 − (0.8² + 0.2²) = 1 − (0.64 + 0.04) = 0.32`.

**Cómo lo usa el algoritmo (ej. CART):** en cada nodo prueba muchas
particiones posibles (distintas features, distintos umbrales), calcula el
**Gini ponderado** de los hijos y **elige la partición con menor Gini**
(la que deja los nodos más puros). Repite recursivamente.
- (Alternativa al Gini: la **Entropía** / *Information Gain*, más teórica.
  Ambas buscan lo mismo: nodos puros.)

### El problema del árbol: Overfitting
Si el árbol **crece sin límite**, divide hasta que cada hoja sea pura
(Gini = 0). Eso produce un árbol **muy profundo que memoriza** el
entrenamiento → overfitting.

**Cómo evitarlo:**
- **Pre-pruning (poda temprana, lo más usado):** poner límites con
  hiperparámetros:
  - `max_depth` → profundidad máxima.
  - `min_samples_split` → mínimo de datos para dividir un nodo.
  - `min_samples_leaf` → mínimo de datos en una hoja.
  - `max_leaf_nodes`, `min_impurity_decrease`.
- **Post-pruning (poda posterior):** dejar crecer el árbol y luego recortar
  ramas inútiles (en sklearn, `ccp_alpha`).

### Árboles para regresión
En lugar de Gini, usan el **Error Cuadrático Medio (MSE)**. La predicción de
una hoja es el **promedio** de los valores de los ejemplos que caen ahí.

### ⭐ Ventaja estrella: NO necesitan escalado
> "Los árboles deciden con **umbrales sobre una feature a la vez**. Escalar
> (multiplicar/sumar una constante) no cambia qué umbral es el mejor."
>
> (En el Notebook 12 el F1 con y sin escalado fue idéntico: 0.933.)
> Por eso son diferentes de SVM y K-Means, que **sí** dependen de distancias.

### Limitaciones (que llevan al Random Forest)
- **Overfitting** fácil.
- **Inestabilidad:** un pequeño cambio en los datos puede dar un árbol muy distinto.
- **Óptimos locales:** el enfoque "voraz" no garantiza el mejor árbol global.

---

## 11. Random Forest y Ensemble Learning

> **Caso práctico:** Notebook 13 — *Detección de malware en Android* con
> Random Forest (mismo dataset que el árbol, para compararlos).

### Ensemble Learning (aprendizaje conjunto)
> "Combinar **muchos modelos** suele dar una predicción **mejor** que cualquier
> modelo individual." (Como pedir opinión a un comité en vez de a una sola persona.)

- Para clasificación: **votación mayoritaria** (la clase más repetida).
- Métodos principales: **bagging, pasting, boosting y stacking**.

**Bagging vs Pasting** (se diferencian en el muestreo):
- **Bagging:** cada modelo se entrena con un subconjunto donde **un ejemplo
  puede repetirse** (muestreo con reemplazo / *bootstrap*).
- **Pasting:** **sin repetición** dentro del subconjunto.

### Random Forest = bagging de árboles
> "Random Forest entrena **muchos árboles de decisión**, cada uno con un
> subconjunto aleatorio de datos (bagging), y combina sus predicciones por
> **votación** (clasificación) o **promedio** (regresión)."

**Doble aleatoriedad (lo que lo hace especial):**
1. Cada árbol ve un subconjunto **aleatorio de datos** (bootstrap).
2. En **cada nodo** solo considera un subconjunto **aleatorio de features**
   (no todas), p. ej. √(nº features).

→ Esto hace que los árboles sean **distintos entre sí** (menos correlacionados).
Sus errores son independientes y **tienden a cancelarse** → modelo de **menor
varianza y más robusto** que un árbol solo.

**Hiperparámetros clave:**
- `n_estimators` → número de árboles (ej. 100).
- `max_depth` → profundidad de cada árbol.
- `n_jobs=-1` → usa todos los núcleos del PC (entrena los árboles en paralelo,
  importante con datasets grandes).

**Resultado del caso práctico (Notebook 13):**
| Modelo | F1 Train | F1 Validación |
|---|---|---|
| Árbol individual (sin límite) | 0.981 | 0.930 |
| Random Forest | 0.981 | **0.933** |

La mejora parece pequeña (+0.003) pero el RF es **más estable y robusto**
ante datos nuevos (menor varianza).

**¿Qué se gana y qué se pierde frente a un árbol?**
- ✅ Se gana: precisión, estabilidad, resistencia al overfitting.
- ❌ Se pierde: **interpretabilidad** (100 árboles no se pueden leer como uno solo).

> Igual que los árboles, **Random Forest NO necesita escalado** (hereda la
> invarianza al escalado).

---

## 12. Selección y Extracción de características

Ambas técnicas reducen el **número de dimensiones** (features), pero de forma
distinta. **No confundirlas** (pregunta típica).

### Validación cruzada (Cross-Validation) — relacionada
> "En lugar de evaluar con una sola división, **se evalúa varias veces con
> distintas divisiones** de los datos."

- `cv=5` → divide el train en **5 partes (folds)**; entrena 5 veces (4 folds
  para entrenar, 1 para validar) **rotando** cuál es la de validación.
- Evita que la evaluación dependa de una partición "con suerte" y ayuda a
  **elegir hiperparámetros fiables**.
- El **Test Set se reserva** y solo se usa al final.

### Selección de características (Feature Selection)
> "**Elimina** columnas (features) poco útiles, **manteniendo el resto igual**."

Objetivo: reducir dimensionalidad → mejor rendimiento, datos más limpios,
modelo más interpretable, menos overfitting.

**¿Cómo se decide qué feature es relevante?**
- **Impacto en el rendimiento:** quito una feature, reentreno; si el
  rendimiento **cae mucho**, esa feature era relevante.
- **Correlación con la salida:** a más correlación con la etiqueta, más relevante.
- **Algoritmos que dan importancia:** Random Forest mide la importancia de cada
  feature (cuánto reduce la impureza). Sirve para descartar las de importancia ≈ 0.

### Extracción de características (Feature Extraction)
> "**Transforma** las features en otras **nuevas y distintas** (menos en
> número). Las nuevas features ya no son las originales."

- **Diferencia clave con la selección:** la selección **conserva** algunas
  columnas tal cual; la extracción **crea columnas nuevas** combinando las
  viejas.
- Algoritmos clave: **PCA (Análisis de Componentes Principales)** y
  **SVD (Descomposición en Valores Singulares)**. PCA usa SVD para proyectar
  los datos a menos dimensiones **maximizando la varianza**.
- Bonus: como las nuevas features no se parecen a las originales, sirve para
  **ocultar/anonimizar** datos sensibles (ej. datos bancarios).

| | Selección | Extracción |
|---|---|---|
| Qué hace | borra columnas | crea columnas nuevas |
| ¿Conserva las originales? | sí (las que quedan) | no |
| Ejemplos | correlación, importancia de RF | PCA, SVD |

---

## 13. Clustering (K-Means y DBSCAN)

> **Caso práctico:** Notebook 17 — *Detección de transacciones bancarias
> fraudulentas* con K-Means. **El único notebook NO supervisado.**

### Idea central
> "El clustering **agrupa datos SIN etiqueta** según su **proximidad/similitud**.
> El objetivo es **dar estructura** a los datos y descubrir patrones, no predecir."

- Es **aprendizaje no supervisado**: el modelo **nunca ve las etiquetas** al
  entrenar. (En el Notebook 17 las etiquetas de "fraude" solo se usaron al
  final, para *evaluar* los grupos.)
- **Clasificar vs Agrupar:** clasificar = asignar una etiqueta **conocida**
  (supervisado). Agrupar = descubrir grupos **sin etiquetas previas**.

### K-Means
> "K-Means agrupa los datos en **K grupos** alrededor de **K centroides**,
> asignando cada punto al centroide más cercano."

**Funcionamiento (4 pasos, te lo pueden pedir):**
1. **Inicializar** K centroides al azar.
2. **Asignar** cada punto al centroide **más cercano** (distancia euclidiana).
3. **Mover** cada centroide a la **media** de los puntos de su grupo.
4. **Repetir** 2 y 3 hasta que los centroides **dejen de moverse** (convergencia).

Minimiza la **suma de distancias al cuadrado** entre cada punto y el centroide
de su cluster.

**Limitaciones (importantes):**
- **Hay que elegir K de antemano** (K es un **hiperparámetro**).
- **Necesita escalado** obligatorio (usa distancias euclidianas: una feature de
  rango enorme dominaría el cálculo).
- **No sirve para categóricas con One-Hot** (interpretaría falsas distancias).
- Pierde eficacia con **muchas dimensiones** (conviene reducir con PCA/SVD).
- **Sensible a la inicialización** de centroides.
- **Asume clusters esféricos** → falla con formas raras (anulares, alargadas).

### DBSCAN (alternativa basada en densidad)
> "DBSCAN agrupa por **densidad**: busca zonas donde haya muchos puntos juntos,
> en lugar de basarse en la distancia a un centroide como K-Means."

- **Ventajas:** maneja clusters **no esféricos** e **infiere el número de
  clusters automáticamente** (no hay que decirle K).
- **Parámetros clave:** `epsilon` (radio de vecindad) y `min_points`
  (vecinos mínimos).
- **Tres tipos de puntos:**
  - **Core point:** tiene al menos `min_points` vecinos dentro de su radio.
  - **Border point:** no es core, pero está cerca de un core point.
  - **Noise point:** ni core ni border → se considera **anomalía** (no entra
    en ningún cluster).
- **Limitaciones:** falla con clusters de **densidades muy distintas**, muy
  dependiente de elegir bien `epsilon`/`min_points`, y sufre en alta dimensión.

### Cómo se evalúa un clustering
- **Si HAY etiquetas (para validar):** homogeneidad, plenitud (completitud),
  **V-Measure** (combina ambas, como el F1) y **pureza** (asignar cada cluster
  a su clase mayoritaria y medir aciertos).
  - ⚠️ La **pureza engaña con datos desbalanceados**: si casi todo es
    "legítimo", saldrá altísima aunque el clustering sea malo.
- **Si NO hay etiquetas:**
  - **Coeficiente de Silhouette:** mide cohesión interna vs separación entre
    clusters. Va de **-1 a 1**; cerca de 1 = buena agrupación; cerca de 0 =
    clusters solapados. Bueno para K-Means (basado en distancia).
  - **Índice de Calinski-Harabasz:** ratio dispersión entre-grupos / dentro-grupo.
    Más alto = clusters más densos y separados. Útil para DBSCAN. (No tiene
    rango fijo, hay que comparar valores.)

### Truco interesante del caso práctico (Notebook 17)
Se usó un modelo **supervisado (Random Forest)** para elegir las 7 features más
importantes, y luego K-Means con esas 7 features → un cluster concentró el
**54% de todos los fraudes** en solo 308 muestras. *Combinar un modelo
supervisado como auxiliar puede mejorar mucho uno no supervisado.*

---

## 14. Tabla comparativa de algoritmos

| Aspecto | SVM | Árbol de Decisión | Random Forest | K-Means |
|---|---|---|---|---|
| **Aprendizaje** | Supervisado | Supervisado | Supervisado | **No supervisado** |
| **Tipo de problema** | Clasificación (URLs) | Clasif. multiclase (malware) | Clasif. multiclase (malware) | Clustering (fraude) |
| **¿Necesita escalado?** | Sí (obligatorio en RBF) | **No** | **No** | Sí (obligatorio) |
| **Hiperparámetros** | C, kernel, gamma | max_depth, criterion | n_estimators, max_depth | K (nº clusters) |
| **Métricas** | F1, precisión, recall | F1 weighted, matriz conf. | F1 weighted, matriz conf. | Silhouette, Calinski-H., Pureza |
| **Ventaja** | Buen margen, alta dim. | **Interpretable**, rápido | Robusto, preciso | Descubre patrones sin etiquetas |
| **Limitación** | Lento en datasets grandes | Overfitting fácil | Poco interpretable | Hay que fijar K, sensible a escala |

**Reflexión para cerrar (frase de cierre potente para la oral):**
> "Random Forest suele ser el más equilibrado para seguridad informática:
> alta precisión, robusto al overfitting y maneja datasets grandes y
> desbalanceados sin escalar. Su única pega es la menor interpretabilidad
> frente a un árbol individual, pero se compensa mirando la importancia de
> las features."

---

## 15. Errores comunes que debes EVITAR decir

1. ❌ "Hay overfitting cuando acc_train ≪ acc_val."
   ✅ **Es al revés: overfitting cuando acc_train ≫ acc_val** (entrenamiento
   mucho MEJOR que validación). *(Lo tenías mal en el cuestionario.)*

2. ❌ "La etiqueta no es una columna del dataset."
   ✅ **La etiqueta SÍ es una columna**, la columna objetivo (la Y). Las demás
   son las features.

3. ❌ "La regresión logística sirve para predecir números."
   ✅ Sirve para **clasificar** (predice probabilidad → clase 0/1). La que
   predice números es la **lineal**.

4. ❌ "Los árboles y Random Forest necesitan escalado."
   ✅ **NO lo necesitan** (deciden por umbrales). Quienes lo necesitan son
   **SVM (RBF) y K-Means** (usan distancias).

5. ❌ "Hard margin y soft margin son dos algoritmos distintos."
   ✅ Es el **mismo SVM**, regulado por el hiperparámetro **C**.

6. ❌ "Selección y extracción de características es lo mismo."
   ✅ Selección **borra** columnas existentes; extracción **crea columnas
   nuevas** (PCA/SVD).

7. ❌ "K-Means descubre solo cuántos grupos hay."
   ✅ A K-Means **hay que darle K**. El que infiere el número de clusters es
   **DBSCAN**.

8. ❌ "Un valor de C alto en SVM generaliza mejor."
   ✅ C alto = poca tolerancia a errores = **riesgo de overfitting**. C bajo
   regulariza más (modelo más general).

---

## 16. Posibles preguntas orales

**P: ¿Diferencia entre supervisado y no supervisado?**
R: En supervisado los datos tienen etiqueta (Y) y el modelo aprende a predecir
(SVM, árboles, RF). En no supervisado no hay etiqueta y el modelo agrupa o
encuentra patrones solo (K-Means).

**P: ¿Por qué dividimos en train/validation/test?**
R: Train para aprender los parámetros, validation para ajustar hiperparámetros
y elegir el mejor modelo, y test para una evaluación final con datos nunca
vistos. Sin esa separación el modelo parecería bueno pero fallaría en producción.

**P: ¿Qué es overfitting y cómo lo detectas?**
R: Es cuando el modelo memoriza el entrenamiento y falla con datos nuevos. Se
detecta porque el rendimiento en entrenamiento es mucho mejor que en validación
(acc_train ≫ acc_val).

**P: ¿Para qué sirve la función sigmoide?**
R: Transforma la salida lineal (cualquier número) en una probabilidad entre 0 y
1, lo que permite usar la regresión logística para clasificar.

**P: ¿Qué hace el parámetro C en SVM?**
R: Equilibra entre maximizar el margen y minimizar errores. C bajo → margen
amplio, modelo general. C alto → pocos errores permitidos, riesgo de overfitting.

**P: ¿Por qué un árbol no necesita escalado pero K-Means sí?**
R: El árbol decide con umbrales sobre cada feature por separado, y escalar no
cambia el mejor umbral. K-Means usa distancias euclidianas, así que una feature
con rango grande dominaría el cálculo si no se escala.

**P: ¿Por qué Random Forest mejora a un árbol solo?**
R: Combina muchos árboles entrenados con datos y features aleatorios. Sus
errores son independientes y se cancelan al votar, dando un modelo de menor
varianza y más robusto.

**P: ¿Qué es la impureza de Gini?**
R: Mide cuán mezcladas están las clases en un nodo. 0 = nodo puro (una sola
clase), cercano a 0.5 = muy mezclado. El árbol elige las divisiones que
minimizan el Gini.

**P: ¿Qué métrica usarías con clases desbalanceadas y por qué?**
R: F1-Score (o la curva PR), no la accuracy. Con desbalance, la accuracy engaña
porque predecir siempre la clase mayoritaria ya da un valor alto pero inútil.

**P: ¿Diferencia entre K-Means y DBSCAN?**
R: K-Means agrupa por distancia a centroides y necesita que le des K; asume
clusters esféricos. DBSCAN agrupa por densidad, infiere el número de clusters
solo, maneja formas no esféricas y detecta anomalías (ruido).

---

> **Consejo final para la prueba:** para cada algoritmo ten claras 4 cosas:
> (1) ¿supervisado o no?, (2) ¿clasifica, predice números o agrupa?,
> (3) ¿necesita escalado?, (4) su hiperparámetro estrella. Con eso respondes
> casi cualquier pregunta de comparación.
