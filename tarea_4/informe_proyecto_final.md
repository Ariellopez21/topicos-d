# Tarea 4 — Regresión Logística: Predicción de aceptación de una campaña bancaria

## Notebook

`Regresion_Logistica_Campana_Bancaria.ipynb`

## Dataset

**Archivo requerido:** `bank-additional-full.csv`

Descargar desde UCI ML Repository:
[https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

O bien desde Kaggle:
[https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)

**Pasos:**
1. Descargar y descomprimir el archivo `bank+marketing.zip`.
2. Dentro del zip hay una carpeta `bank-additional/` que contiene `bank-additional-full.csv`.
3. Copiar `bank-additional-full.csv` en esta misma carpeta (`tarea_4/`).

El archivo usa `;` como separador de columnas.

## Descripción del dataset

| Característica | Valor |
|---|---|
| Instancias | 41 188 |
| Features de entrada | 20 (mix numérico y categórico) |
| Variable objetivo | `y` (yes / no) — suscripción a depósito a plazo |
| Separador | `;` |
| Valores faltantes | Codificados como `"unknown"` en columnas categóricas |

**Columnas numéricas:** age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

**Columnas categóricas:** job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome

> Nota: `pdays = 999` indica que el cliente no fue contactado previamente (valor centinela, no un valor real).

## Estructura del notebook

| Sección | Descripción |
|---|---|
| Imports | scikit-learn, pandas, numpy, matplotlib |
| Funciones auxiliares | `train_val_test_split` (60/20/20), `remove_labels`, `evaluate_result`, `plot_lr_decision_boundary` |
| 1. Lectura | Carga del CSV con separador `;` |
| 2. Visualización | head, describe, info, value_counts, nulos, infinitos, scatter age vs duration |
| 3. División | Split estratificado 60/20/20 por etiqueta `y` |
| 4. Preparación | `OrdinalEncoder` para categóricas + `SimpleImputer(median)` |
| 5. LR sin escalado | 5.1 Conjunto reducido (2 features + frontera de decisión) / 5.2 Completo |
| 6. LR con escalado | 6.1 lbfgs + L2 / 6.2 liblinear + L1 / 6.3 Evaluación comparativa |

## Referencia

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31.
