"""
Feature engineering for pairs trading ML models.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def create_features(df: pd.DataFrame, 
                   window: int = 20,
                   lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """
    Crée des features techniques pour le pairs trading
    """
    data = df.copy()
    
    # Features de base
    data['ratio'] = data['Asset_A'] / data['Asset_B']
    data['spread'] = data['Asset_A'] - data['Asset_B']
    data['log_ratio'] = np.log(data['ratio'])
    
    # Moyennes mobiles du ratio
    data['ratio_ma'] = data['ratio'].rolling(window).mean()
    data['ratio_std'] = data['ratio'].rolling(window).std()
    data['zscore'] = (data['ratio'] - data['ratio_ma']) / data['ratio_std']
    
    # Rendements
    data['return_A'] = data['Asset_A'].pct_change()
    data['return_B'] = data['Asset_B'].pct_change()
    data['return_spread'] = data['return_A'] - data['return_B']
    
    # Volatilités
    data['vol_A'] = data['return_A'].rolling(window).std() * np.sqrt(252)
    data['vol_B'] = data['return_B'].rolling(window).std() * np.sqrt(252)
    data['vol_ratio'] = data['vol_A'] / data['vol_B']
    data['vol_spread'] = data['return_spread'].rolling(window).std() * np.sqrt(252)
    
    # Corrélation mobile
    data['corr'] = data['return_A'].rolling(window).corr(data['return_B'])
    
    # Momentum
    data['momentum_A_5'] = data['Asset_A'].pct_change(5)
    data['momentum_A_10'] = data['Asset_A'].pct_change(10)
    data['momentum_A_20'] = data['Asset_A'].pct_change(20)
    data['momentum_B_5'] = data['Asset_B'].pct_change(5)
    data['momentum_B_10'] = data['Asset_B'].pct_change(10)
    data['momentum_B_20'] = data['Asset_B'].pct_change(20)
    
    # RSI simplifié
    data['rsi_A'] = compute_rsi(data['return_A'], window=14)
    data['rsi_B'] = compute_rsi(data['return_B'], window=14)
    data['rsi_spread'] = compute_rsi(data['return_spread'], window=14)
    
    # Distance par rapport aux moyennes mobiles
    data['dist_ma20_A'] = data['Asset_A'] / data['Asset_A'].rolling(20).mean() - 1
    data['dist_ma50_A'] = data['Asset_A'] / data['Asset_A'].rolling(50).mean() - 1
    data['dist_ma20_B'] = data['Asset_B'] / data['Asset_B'].rolling(20).mean() - 1
    data['dist_ma50_B'] = data['Asset_B'] / data['Asset_B'].rolling(50).mean() - 1
    
    # Lagged features
    for lag in lags:
        data[f'ratio_lag_{lag}'] = data['ratio'].shift(lag)
        data[f'zscore_lag_{lag}'] = data['zscore'].shift(lag)
        data[f'return_A_lag_{lag}'] = data['return_A'].shift(lag)
        data[f'return_B_lag_{lag}'] = data['return_B'].shift(lag)
        data[f'vol_A_lag_{lag}'] = data['vol_A'].shift(lag)
        data[f'vol_B_lag_{lag}'] = data['vol_B'].shift(lag)
        data[f'corr_lag_{lag}'] = data['corr'].shift(lag)
    
    return data


def compute_rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    """
    Calcule le RSI (Relative Strength Index) à partir des rendements
    """
    gains = returns.where(returns > 0, 0)
    losses = -returns.where(returns < 0, 0)
    
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def create_labels(df: pd.DataFrame, 
                 horizon: int = 5,
                 zscore_threshold: float = 1.5,
                 return_threshold: float = 0.01) -> pd.DataFrame:
    """
    Crée les labels pour la classification supervisée
    """
    data = df.copy()
    
    # Rendement futur du spread
    data['future_spread_return'] = data['return_spread'].shift(-horizon).rolling(horizon).sum()
    
    # Label = 1 si le spread est extrême ET retourne vers la moyenne
    data['label'] = 0
    
    # Pour position longue (zscore < -threshold)
    long_condition = (data['zscore'] < -zscore_threshold) & (data['future_spread_return'] > return_threshold)
    
    # Pour position courte (zscore > threshold)
    short_condition = (data['zscore'] > zscore_threshold) & (data['future_spread_return'] < -return_threshold)
    
    data.loc[long_condition | short_condition, 'label'] = 1
    
    return data


def get_feature_names() -> List[str]:
    """
    Retourne la liste des noms de features utilisées
    """
    features = [
        'ratio', 'zscore', 'vol_A', 'vol_B', 'vol_ratio', 'corr',
        'momentum_A_5', 'momentum_A_10', 'momentum_A_20',
        'momentum_B_5', 'momentum_B_10', 'momentum_B_20',
        'rsi_A', 'rsi_B', 'rsi_spread',
        'dist_ma20_A', 'dist_ma50_A', 'dist_ma20_B', 'dist_ma50_B'
    ]
    
    for lag in [1, 2, 3, 5]:
        features.extend([
            f'ratio_lag_{lag}',
            f'zscore_lag_{lag}',
            f'return_A_lag_{lag}',
            f'return_B_lag_{lag}'
        ])
    
    return features


def prepare_data_for_ml(df: pd.DataFrame,
                        feature_cols: List[str],
                        target_col: str = 'label',
                        test_size: float = 0.2,
                        val_size: float = 0.1) -> Tuple:
    """
    Prépare les données pour l'entraînement (split temporel)
    """
    df_clean = df[feature_cols + [target_col]].dropna()
    
    n = len(df_clean)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    
    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]
    
    split_dates = {
        'train': df_clean.index[:train_end],
        'val': df_clean.index[train_end:val_end],
        'test': df_clean.index[val_end:]
    }
    
    return X_train, X_val, X_test, y_train, y_val, y_test, split_dates