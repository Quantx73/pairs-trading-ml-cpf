# Pairs Trading avec Machine Learning

## 📋 Description
Ce projet implémente une stratégie de pairs trading améliorée par des méthodes de machine learning, dans le cadre du Certificate in Python for Finance (CPF).

## 🚀 Structure du projet
- `data/` : Données statiques (CSV)
- `notebooks/` : Notebooks Jupyter
- `src/` : Modules Python
  - `baseline.py` : Stratégie traditionnelle (z-score)
  - `models.py` : Modèles ML (Random Forest)

## 📊 Données
Les données sont des prix synthétiques pour deux actifs corrélés (Asset_A et Asset_B), générés pour la période 2014-2024.

## 🎯 Objectifs
1. Implémenter une baseline de pairs trading (z-score)
2. Développer des modèles ML (Random Forest)
3. Comparer les performances

## 🔧 Installation
```bash
pip install -r requirements.txt