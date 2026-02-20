"""
Machine Learning models for pairs trading.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')


def train_random_forest(X_train, y_train, 
                        X_val=None, y_val=None,
                        n_estimators: int = 100,
                        max_depth: int = 10,
                        random_state: int = 42) -> RandomForestClassifier:
    """
    Entraîne un modèle Random Forest
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Validation si fournie
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"  Validation accuracy: {val_acc:.4f}")
    
    return model


def evaluate_model(model, X_test, y_test) -> Dict:
    """
    Évalue le modèle et retourne les métriques
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        metrics['feature_importance'] = model.feature_importances_
    
    return metrics


def backtest_ml_strategy(df: pd.DataFrame,
                         model,
                         feature_cols: list,
                         target_col: str = 'label',
                         transaction_cost: float = 0.001,
                         zscore_threshold: float = 1.5) -> pd.DataFrame:
    """
    Backtest de la stratégie ML avec signaux de trading
    """
    df = df.copy()
    
    # Préparer les données pour les prédictions
    df_clean = df[feature_cols].dropna()
    
    # Prédire les probabilités
    probabilities = model.predict_proba(df_clean)[:, 1]
    
    # Aligner les index
    df.loc[df_clean.index, 'ml_probability'] = probabilities
    df.loc[df_clean.index, 'ml_prediction'] = (probabilities > 0.5).astype(int)
    
    # Générer les signaux de trading
    df['ml_signal'] = 0
    
    # Long signal (zscore < -threshold ET prédiction = 1)
    long_condition = (df['zscore'] < -zscore_threshold) & (df['ml_prediction'] == 1)
    
    # Short signal (zscore > threshold ET prédiction = 1)
    short_condition = (df['zscore'] > zscore_threshold) & (df['ml_prediction'] == 1)
    
    df.loc[long_condition, 'ml_signal'] = 1
    df.loc[short_condition, 'ml_signal'] = -1
    
    # Positions (avec mémoire)
    df['ml_position'] = 0
    position = 0
    
    for i in range(len(df)):
        current_signal = df.iloc[i]['ml_signal']
        
        if position == 0 and current_signal != 0:
            position = current_signal
        elif position != 0 and current_signal == -position:
            position = 0
        elif position != 0 and current_signal == position:
            pass
            
        df.iloc[i, df.columns.get_loc('ml_position')] = position
    
    # Calcul des rendements
    if 'return_A' not in df.columns:
        df['return_A'] = df['Asset_A'].pct_change()
        df['return_B'] = df['Asset_B'].pct_change()
    
    df['ml_return'] = df['ml_position'] * (df['return_A'] - df['return_B'])
    
    # Coûts de transaction
    df['position_change'] = df['ml_position'].diff().abs() / 2
    df['ml_return_net'] = df['ml_return'] - (df['position_change'] * transaction_cost)
    
    # Rendements cumulés
    df['ml_cumulative_return'] = (1 + df['ml_return'].fillna(0)).cumprod()
    df['ml_cumulative_return_net'] = (1 + df['ml_return_net'].fillna(0)).cumprod()
    
    return df


def run_ml_pipeline(data: pd.DataFrame,
                   window: int = 20,
                   horizon: int = 5,
                   test_size: float = 0.2,
                   rf_params: Optional[Dict] = None) -> Tuple[Any, Dict, pd.DataFrame]:
    """
    Pipeline complet pour le modèle ML
    """
    from src.features import create_features, create_labels, get_feature_names, prepare_data_for_ml
    
    print("=" * 60)
    print("MACHINE LEARNING PIPELINE")
    print("=" * 60)
    
    # 1. Créer les features
    print("\n1. Création des features...")
    df = create_features(data, window=window)
    df = create_labels(df, horizon=horizon)
    
    # 2. Définir les features
    feature_cols = get_feature_names()
    print(f"   {len(feature_cols)} features créées")
    
    # 3. Préparer les données
    print("\n2. Préparation des données...")
    X_train, X_val, X_test, y_train, y_val, y_test, split_dates = prepare_data_for_ml(
        df, feature_cols, target_col='label', test_size=test_size, val_size=0.1
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Classes: 0={sum(y_train==0)}, 1={sum(y_train==1)}")
    
    # 4. Entraîner le modèle
    print("\n3. Entraînement du modèle...")
    if rf_params is None:
        rf_params = {'n_estimators': 100, 'max_depth': 10}
    
    model = train_random_forest(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        **rf_params
    )
    
    # 5. Évaluation
    print("\n4. Évaluation sur le test set...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision (classe 1): {metrics['classification_report']['1']['precision']:.4f}")
    print(f"   Recall (classe 1): {metrics['classification_report']['1']['recall']:.4f}")
    print(f"   F1-score (classe 1): {metrics['classification_report']['1']['f1-score']:.4f}")
    
    # 6. Backtest
    print("\n5. Backtest de la stratégie...")
    df_results = backtest_ml_strategy(df, model, feature_cols)
    
    # Ajouter les dates de split pour référence
    df_results['split'] = 'train'
    df_results.loc[split_dates['val'], 'split'] = 'val'
    df_results.loc[split_dates['test'], 'split'] = 'test'
    
    print("\n✅ Pipeline terminé!")
    
    return model, metrics, df_results


def save_model(model, filename: str = 'random_forest_model.pkl'):
    """
    Sauvegarde le modèle entraîné
    """
    model_path = Path('../models') / filename
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé: {model_path}")


def load_model(filename: str = 'random_forest_model.pkl'):
    """
    Charge un modèle sauvegardé
    """
    model_path = Path('../models') / filename
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"✅ Modèle chargé: {model_path}")
        return model
    else:
        print(f"❌ Fichier non trouvé: {model_path}")
        return None