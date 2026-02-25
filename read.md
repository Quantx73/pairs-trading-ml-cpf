# Pairs Trading with Machine Learning

## 📋 Description
This project implements a pairs trading strategy enhanced with machine learning, as part of the Certificate in Python for Finance (CPF).

## 🚀 Project Structure

### 📂 Folder Details

- **`data/`** : Static CSV files with synthetic price data for two correlated assets
- **`notebooks/`** : Main Jupyter notebook (`Final_Project_Pairs_Trading_ML.ipynb`)
- **`src/`** : Python modules
  - `baseline.py` : Traditional z-score strategy implementation
  - `features.py` : Feature engineering functions for ML models
  - `models.py` : Random Forest model and ML pipeline
  - `backtest.py` : Backtesting utilities and performance metrics


## 📊 Data
The project uses **synthetic data** generated specifically for pairs trading:
- Two highly correlated assets
- Temporary spread deviations with mean reversion
- Realistic long-term trends

This ensures full reproducibility and compliance with CPF requirements (static data files).

## 🎯 Objectives
1. Implement a baseline pairs trading strategy (z-score)
2. Engineer relevant features for machine learning
3. Train a Random Forest model to predict mean reversion
4. Compare performance between traditional and ML approaches

## 🔧 Installation
```bash
pip install -r requirements.txt