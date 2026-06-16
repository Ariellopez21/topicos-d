# Tarea 4 — Regresión Logística y Random Forest: Predicción de aceptación de una campaña bancaria

## Notebook

`Regresion_Logistica_Campana_Bancaria.ipynb`

## Dataset

**Archivo requerido:** `bank-additional-full.csv` (ya está en esta carpeta)

Si necesitas descargarlo de nuevo:
[https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

Descomprimir `bank+marketing.zip` → entrar en `bank-additional/` → copiar `bank-additional-full.csv` aquí.

El archivo usa `;` como separador de columnas.

## Descripción del dataset

| Característica | Valor |
|---|---|
| Instancias | 41 188 |
| Features de entrada | 20 (10 numéricas + 10 categóricas) |
| Variable objetivo | `y` (yes / no) — suscripción a depósito a plazo |
| Separador | `;` |
| Desbalance | ≈ 88 % `no` / ≈ 12 % `yes` |

**Columnas numéricas:** age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

**Columnas categóricas:** job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome

## Algoritmos

| Algoritmo | Tipo | Hiperparámetros explorados |
|---|---|---|
| Regresión Logística | Modelo lineal simple | `C` |
| Random Forest | Modelo de ensamble | `n_estimators`, `max_depth` |

## Estructura del notebook

| Sección | Descripción |
|---|---|
| Imports | scikit-learn, pandas, numpy, matplotlib |
| Funciones auxiliares | `train_val_test_split` (60/20/20), `remove_labels`, `evaluate_result`, `plot_lr_decision_boundary` |
| 1. Lectura | Carga del CSV con separador `;` |
| 2. Visualización | head, describe, info, desbalance de clases, nulos, infinitos, scatter |
| 3. División | Split estratificado 60/20/20 por etiqueta `y` |
| 4. Preparación | `OneHotEncoder` para categóricas + `SimpleImputer(median)` |
| 5–6. Regresión Logística | Sin/con escalado, L1 vs L2, exploración de `C` |
| 7. Random Forest | Modelo base, exploración de `n_estimators` y `max_depth`, importancia de features |
| 8. Comparación | `classification_report` y F1 Score LR vs RF |

## Referencia

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.
