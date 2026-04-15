# Informe — Regresión Lineal: Coste de un incidente de seguridad

## Objetivo

Predecir el coste económico de un incidente de ciberseguridad en función del número de equipos afectados, usando un modelo de regresión lineal simple entrenado sobre datos sintéticos.

---

## 1. Generación del conjunto de datos

```python
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```

Se generan 100 muestras de forma aleatoria siguiendo la función lineal **y = 4 + 3X + ruido gaussiano**. `X` representa el número de equipos afectados (en escala normalizada, rango [0, 2]) e `y` el coste del incidente. El ruido (`randn`) simula la variabilidad real de un incidente.

---

## 2. Construcción del DataFrame y escalado

```python
data = {'n_equipos_afectados': X.flatten(), 'coste': y.flatten()}
df = pd.DataFrame(data)
```

Los arrays de NumPy se convierten a un DataFrame de Pandas para facilitar su manipulación.

```python
df['n_equipos_afectados'] = (df['n_equipos_afectados'] * 1000).astype('int')
df['coste'] = (df['coste'] * 10000).astype('int')
```

Las columnas se reescalan a unidades interpretables: el número de equipos pasa a un rango **[0 – 2000]** y el coste a **[~40.000 – ~110.000 €]**. Esto no altera la relación matemática entre variables, solo da sentido semántico a los valores.

---

## 3. Entrenamiento del modelo

```python
lin_reg = LinearRegression()
lin_reg.fit(df['n_equipos_afectados'].values.reshape(-1, 1), df['coste'].values)
```

Se entrena un modelo de regresión lineal simple de scikit-learn. El `.reshape(-1, 1)` es necesario porque `fit()` espera una matriz 2D para `X`. El modelo ajusta internamente los parámetros θ₀ y θ₁ minimizando el error cuadrático medio (OLS).

---

## 4. Parámetros del modelo aprendidos

```python
lin_reg.intercept_   # → ~39144
lin_reg.coef_        # → [~31.84]
```

La función hipótesis resultante es:

> **coste = 39.144 + 31,84 × n_equipos_afectados**

- **θ₀ ≈ 39.144 €** — coste base del incidente aunque no haya ningún equipo afectado (costes fijos de respuesta).
- **θ₁ ≈ 31,84 €/equipo** — incremento del coste por cada equipo adicional comprometido.

---

## 5. Preparación de la línea de predicción para el gráfico

```python
X_min_max = np.array([[df["n_equipos_afectados"].min()], [df["n_equipos_afectados"].max()]])
y_train_pred = lin_reg.predict(X_min_max)
```

Para dibujar la recta de regresión solo se necesitan dos puntos: el correspondiente al valor mínimo y al máximo del conjunto de entrenamiento. Se predice el coste para ambos extremos y se traza la línea entre ellos.

---

## 6. Predicción de un nuevo ejemplo

```python
x_new = np.array([[1300]])
coste = lin_reg.predict(x_new)
# → El coste del incidente sería: ~80.539 €
```

Se aplica el modelo a un caso hipotético de **1.300 equipos afectados**. El modelo estima un coste de aproximadamente **80.539 €**, que se puede verificar manualmente con la ecuación aprendida:

> 39.144 + 31,84 × 1.300 ≈ **80.536 €** ✓
