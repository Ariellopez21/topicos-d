# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Machine Learning for Cybersecurity course repository containing two Jupyter notebooks that demonstrate ML techniques applied to security use cases. The notebooks are designed to run both locally and in Google Colab.

## Environment Setup

Dependencies are managed via a local `.venv` (Python 3.12):

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies (also done inline in notebooks)
pip install pandas numpy matplotlib scikit-learn nltk
```

## Running Notebooks

```bash
# Launch Jupyter to work with notebooks
jupyter notebook

# Run a notebook non-interactively (useful for testing)
jupyter nbconvert --to notebook --execute "4_Regresión+Lineal+-+Predicción+del+coste+de+un+incidente+de+seguridad.ipynb"
jupyter nbconvert --to notebook --execute "5_Regresión+Logística+-+Detección+de+SPAM.ipynb"
```

## Notebooks

### Notebook 4 — Linear Regression: Security Incident Cost Prediction
- Generates a synthetic dataset (100 samples) relating number of affected systems to incident cost
- Trains a `sklearn.linear_model.LinearRegression` model
- No external dataset required — data is randomly generated with `numpy`

### Notebook 5 — Logistic Regression: SPAM Detection
- Uses the **TREC 2007 Public Spam Corpus** (`datasets/trec07p/`), which must be present locally
- Email preprocessing pipeline: HTML tag stripping → tokenization → stopword removal → Porter stemming
- Feature extraction via `CountVectorizer`, classification via `LogisticRegression`
- Dataset path expected at `datasets/trec07p/` relative to the notebook

## Architecture Notes

Notebook 5 has a layered preprocessing structure:
1. `MLStripper` (HTMLParser subclass) — strips HTML tags from email bodies
2. `Parser` class — wraps NLTK stemmer/stopwords, parses raw email files into `{subject, body}` dicts
3. `parse_index()` — reads the TREC index file, returns list of `{email_path, label}` dicts
4. `create_prep_dataset()` — combines the above to build `X` (text strings) and `y` (spam/ham labels) arrays ready for `CountVectorizer`
