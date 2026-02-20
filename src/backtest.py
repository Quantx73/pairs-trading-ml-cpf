"""
Backtesting utilities for pairs trading strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional


def calculate_returns(df: pd.DataFrame, 
                      position_col: str = 'position',
                      price_cols: Tuple[str, str] = ('Asset_A', 'Asset_B'),
                      transaction_cost: float = 0.001) -> pd.DataFrame:
    """
    Calcule les rendements d'une stratégie
    """
    df = df.copy()
    
    # Rendements journaliers
    df['return_A'] = df[price_cols[0]].pct_change()
    df['return_B'] = df[price_cols[1]].pct_change()
    
    # Rendement de la stratégie
    df['strategy_return'] = df[position_col] * (df['return_A'] - df['return_B'])
    
    # Coûts de transaction
    df['position_change'] = df[position_col].diff().abs() / 2
    df['strategy_return_net'] = df['strategy_return'] - (df['position_change'] * transaction_cost)
    
    # Rendements cumulés
    df['cumulative_return_gross'] = (1 + df['strategy_return'].fillna(0)).cumprod()
    df['cumulative_return_net'] = (1 + df['strategy_return_net'].fillna(0)).cumprod()
    
    return df


def calculate_metrics(df: pd.DataFrame,
                      return_col: str = 'strategy_return',
                      cumulative_col: str = 'cumulative_return_net') -> Dict:
    """
    Calcule les métriques de performance
    """
    df_clean = df.dropna()
    
    metrics = {}
    
    # Rendements
    metrics['total_return'] = df_clean[cumulative_col].iloc[-1] - 1
    
    # Annualisés
    n_years = len(df_clean) / 252
    metrics['annualized_return'] = (1 + metrics['total_return']) ** (1 / n_years) - 1
    
    # Volatilité
    metrics['annualized_vol'] = df_clean[return_col].std() * np.sqrt(252)
    
    # Sharpe ratio
    metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_vol'] if metrics['annualized_vol'] != 0 else 0
    
    # Maximum drawdown
    rolling_max = df_clean[cumulative_col].expanding().max()
    drawdown = df_clean[cumulative_col] / rolling_max - 1
    metrics['max_drawdown'] = drawdown.min()
    
    # Win rate
    metrics['win_rate'] = (df_clean[return_col] > 0).mean() * 100
    
    # Nombre de trades
    if 'position' in df_clean.columns:
        metrics['num_trades'] = (df_clean['position'].diff() != 0).sum() / 2
        metrics['trades_per_year'] = metrics['num_trades'] / n_years
    
    # Profit factor
    positive_returns = df_clean[df_clean[return_col] > 0][return_col].sum()
    negative_returns = abs(df_clean[df_clean[return_col] < 0][return_col].sum())
    metrics['profit_factor'] = positive_returns / negative_returns if negative_returns != 0 else float('inf')
    
    return metrics


def plot_comparison(baseline_df: pd.DataFrame,
                   ml_df: pd.DataFrame,
                   baseline_col: str = 'cumulative_return_net',
                   ml_col: str = 'ml_cumulative_return_net',
                   baseline_name: str = 'Baseline Z-Score',
                   ml_name: str = 'Random Forest',
                   save_path: Optional[str] = None):
    """
    Compare les courbes de rendement cumulé
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(baseline_df.index, baseline_df[baseline_col], 
            label=baseline_name, linewidth=2, color='blue')
    ax.plot(ml_df.index, ml_df[ml_col], 
            label=ml_name, linewidth=2, color='green')
    
    ax.set_title('Comparaison des rendements cumulés')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rendement cumulé')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_metrics(metrics: Dict, name: str = "Strategy"):
    """
    Affiche les métriques formatées
    """
    print(f"\n{name} Performance Metrics:")
    print("-" * 40)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Vol: {metrics['annualized_vol']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    
    if 'num_trades' in metrics:
        print(f"Number of Trades: {metrics['num_trades']:.0f}")
        print(f"Trades per Year: {metrics['trades_per_year']:.1f}")
    
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")