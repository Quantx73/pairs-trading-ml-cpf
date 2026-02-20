"""
Baseline strategy for pairs trading using z-score method.
This module implements the traditional statistical arbitrage approach.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional


def calculate_spread_and_zscore(data: pd.DataFrame, 
                                window: int = 20,
                                method: str = 'ratio') -> pd.DataFrame:
    """
    Calcule le spread et le z-score pour une paire d'actifs
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame avec les colonnes 'Asset_A' et 'Asset_B'
    window : int
        Fenêtre pour la moyenne mobile et l'écart-type
    method : str
        'ratio' ou 'spread' - méthode de calcul
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec spread et z-score ajoutés
    """
    df = data.copy()
    
    if method == 'ratio':
        # Méthode du ratio des prix
        df['ratio'] = df['Asset_A'] / df['Asset_B']
        df['ratio_ma'] = df['ratio'].rolling(window=window).mean()
        df['ratio_std'] = df['ratio'].rolling(window=window).std()
        df['zscore'] = (df['ratio'] - df['ratio_ma']) / df['ratio_std']
        df['spread'] = df['Asset_A'] - df['ratio_ma'] * df['Asset_B']
        
    else:
        # Méthode du spread simple
        df['spread'] = df['Asset_A'] - df['Asset_B']
        df['spread_ma'] = df['spread'].rolling(window=window).mean()
        df['spread_std'] = df['spread'].rolling(window=window).std()
        df['zscore'] = (df['spread'] - df['spread_ma']) / df['spread_std']
    
    return df


def generate_signals(df: pd.DataFrame, 
                     entry_threshold: float = 2.0, 
                     exit_threshold: float = 0.5,
                     stop_loss: Optional[float] = None) -> pd.DataFrame:
    """
    Génère les signaux d'entrée/sortie basés sur le z-score
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec colonne 'zscore'
    entry_threshold : float
        Seuil d'entrée (quand ouvrir une position)
    exit_threshold : float
        Seuil de sortie (quand fermer la position)
    stop_loss : float, optional
        Stop loss en unités de z-score
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec signaux ajoutés
    """
    df = df.copy()
    
    # Initialiser les positions
    df['position'] = 0
    df['signal'] = 0
    
    # Signaux d'entrée
    df.loc[df['zscore'] < -entry_threshold, 'signal'] = 1   # Long signal
    df.loc[df['zscore'] > entry_threshold, 'signal'] = -1   # Short signal
    
    # Simulation des positions avec mémoire
    position = 0
    entry_zscore = None
    
    for i in range(len(df)):
        current_zscore = df.iloc[i]['zscore']
        current_signal = df.iloc[i]['signal']
        
        # Pas de position
        if position == 0:
            if current_signal == 1:
                position = 1
                entry_zscore = current_zscore
            elif current_signal == -1:
                position = -1
                entry_zscore = current_zscore
        
        # Position longue
        elif position == 1:
            # Conditions de sortie
            exit_condition = (abs(current_zscore) < exit_threshold)
            
            # Stop loss si défini
            if stop_loss is not None:
                stop_condition = (current_zscore > entry_zscore + stop_loss)
                exit_condition = exit_condition or stop_condition
            
            if exit_condition or current_signal == -1:
                position = 0
                entry_zscore = None
        
        # Position courte
        elif position == -1:
            # Conditions de sortie
            exit_condition = (abs(current_zscore) < exit_threshold)
            
            # Stop loss si défini
            if stop_loss is not None:
                stop_condition = (current_zscore < entry_zscore - stop_loss)
                exit_condition = exit_condition or stop_condition
            
            if exit_condition or current_signal == 1:
                position = 0
                entry_zscore = None
        
        df.iloc[i, df.columns.get_loc('position')] = position
    
    return df


def calculate_returns(df: pd.DataFrame, 
                      transaction_cost: float = 0.001) -> pd.DataFrame:
    """
    Calcule les rendements de la stratégie
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec colonnes 'Asset_A', 'Asset_B', 'position'
    transaction_cost : float
        Coût de transaction en pourcentage (ex: 0.001 = 0.1%)
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec rendements ajoutés
    """
    df = df.copy()
    
    # Rendements journaliers
    df['return_A'] = df['Asset_A'].pct_change()
    df['return_B'] = df['Asset_B'].pct_change()
    
    # Rendement du spread (long A, short B)
    df['strategy_return'] = df['position'] * (df['return_A'] - df['return_B'])
    
    # Identifier les changements de position (trades)
    df['position_change'] = df['position'].diff().abs() / 2  # Divisé par 2 car changement de +/-1 à 0 compte pour 1 trade
    
    # Appliquer les coûts de transaction
    df['strategy_return_net'] = df['strategy_return'] - (df['position_change'] * transaction_cost)
    
    # Rendements cumulés
    df['cumulative_return_gross'] = (1 + df['strategy_return']).cumprod()
    df['cumulative_return_net'] = (1 + df['strategy_return_net']).cumprod()
    
    # Buy & Hold returns (pour comparaison)
    df['bh_return_A'] = (1 + df['return_A']).cumprod()
    df['bh_return_B'] = (1 + df['return_B']).cumprod()
    
    return df


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """
    Calcule les métriques de performance
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec colonnes de rendements
    
    Returns
    -------
    Dict
        Dictionnaire avec les métriques
    """
    df_clean = df.dropna()
    
    metrics = {}
    
    # Rendements
    metrics['total_return_gross'] = df_clean['cumulative_return_gross'].iloc[-1] - 1
    metrics['total_return_net'] = df_clean['cumulative_return_net'].iloc[-1] - 1
    
    # Annualisés
    n_years = len(df_clean) / 252
    metrics['annualized_return_gross'] = (1 + metrics['total_return_gross']) ** (1 / n_years) - 1
    metrics['annualized_return_net'] = (1 + metrics['total_return_net']) ** (1 / n_years) - 1
    
    # Volatilité
    metrics['annualized_vol'] = df_clean['strategy_return'].std() * np.sqrt(252)
    
    # Sharpe ratio (supposant risk-free rate = 0)
    metrics['sharpe_ratio_gross'] = metrics['annualized_return_gross'] / metrics['annualized_vol'] if metrics['annualized_vol'] != 0 else 0
    metrics['sharpe_ratio_net'] = metrics['annualized_return_net'] / metrics['annualized_vol'] if metrics['annualized_vol'] != 0 else 0
    
    # Maximum drawdown
    rolling_max = df_clean['cumulative_return_gross'].expanding().max()
    drawdown = df_clean['cumulative_return_gross'] / rolling_max - 1
    metrics['max_drawdown'] = drawdown.min()
    
    # Win rate
    metrics['win_rate'] = (df_clean['strategy_return'] > 0).mean() * 100
    
    # Nombre de trades
    metrics['num_trades'] = (df_clean['position'].diff() != 0).sum() / 2
    metrics['trades_per_year'] = metrics['num_trades'] / n_years
    
    # Profit factor
    positive_returns = df_clean[df_clean['strategy_return'] > 0]['strategy_return'].sum()
    negative_returns = abs(df_clean[df_clean['strategy_return'] < 0]['strategy_return'].sum())
    metrics['profit_factor'] = positive_returns / negative_returns if negative_returns != 0 else float('inf')
    
    return metrics


def plot_results(df: pd.DataFrame, 
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 save_path: Optional[str] = None):
    """
    Visualise les résultats de la stratégie
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec résultats
    entry_threshold : float
        Seuil d'entrée (pour visualisation)
    exit_threshold : float
        Seuil de sortie (pour visualisation)
    save_path : str, optional
        Chemin pour sauvegarder le graphique
    """
    df_clean = df.dropna()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Prix des actifs
    axes[0].plot(df_clean.index, df_clean['Asset_A'], label='Asset A', alpha=0.8)
    axes[0].plot(df_clean.index, df_clean['Asset_B'], label='Asset B', alpha=0.8)
    axes[0].set_title('Prix des actifs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Z-score et signaux
    axes[1].plot(df_clean.index, df_clean['zscore'], label='Z-score', color='blue', alpha=0.7)
    axes[1].axhline(y=entry_threshold, color='red', linestyle='--', 
                    label=f'+{entry_threshold} (short entry)')
    axes[1].axhline(y=-entry_threshold, color='green', linestyle='--', 
                    label=f'-{entry_threshold} (long entry)')
    axes[1].axhline(y=exit_threshold, color='orange', linestyle=':', 
                    label=f'Exit ±{exit_threshold}')
    axes[1].axhline(y=-exit_threshold, color='orange', linestyle=':')
    axes[1].fill_between(df_clean.index, -entry_threshold, entry_threshold, 
                         alpha=0.1, color='gray')
    axes[1].set_title('Z-score du ratio et signaux de trading')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Marquer les points d'entrée
    entries_long = df_clean[df_clean['signal'] == 1].index
    entries_short = df_clean[df_clean['signal'] == -1].index
    axes[1].scatter(entries_long, df_clean.loc[entries_long, 'zscore'], 
                   color='green', marker='^', s=50, label='Long signals', zorder=5)
    axes[1].scatter(entries_short, df_clean.loc[entries_short, 'zscore'], 
                   color='red', marker='v', s=50, label='Short signals', zorder=5)
    
    # 3. Positions et rendements
    ax2_twin = axes[2].twinx()
    
    # Positions
    axes[2].plot(df_clean.index, df_clean['position'], 
                label='Position (1=long spread, -1=short spread)', 
                color='purple', drawstyle='steps-post', alpha=0.7)
    axes[2].set_ylabel('Position', color='purple')
    axes[2].tick_params(axis='y', labelcolor='purple')
    axes[2].set_ylim(-1.5, 1.5)
    
    # Rendements cumulés
    ax2_twin.plot(df_clean.index, df_clean['cumulative_return_gross'], 
                  label='Rendement brut', color='darkgreen', linewidth=2)
    ax2_twin.plot(df_clean.index, df_clean['cumulative_return_net'], 
                  label='Rendement net (frais)', color='orange', linewidth=2, alpha=0.7)
    ax2_twin.set_ylabel('Rendement cumulé', color='darkgreen')
    ax2_twin.tick_params(axis='y', labelcolor='darkgreen')
    
    # Buy & Hold pour comparaison
    ax2_twin.plot(df_clean.index, df_clean['bh_return_A'], 
                  label='Buy & Hold A', color='blue', linewidth=1, alpha=0.5, linestyle='--')
    ax2_twin.plot(df_clean.index, df_clean['bh_return_B'], 
                  label='Buy & Hold B', color='red', linewidth=1, alpha=0.5, linestyle='--')
    
    axes[2].set_title('Positions et rendements cumulés')
    axes[2].grid(True, alpha=0.3)
    
    # Légendes combinées
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def run_baseline_strategy(data: pd.DataFrame,
                          window: int = 20,
                          entry_threshold: float = 2.0,
                          exit_threshold: float = 0.5,
                          stop_loss: Optional[float] = None,
                          transaction_cost: float = 0.001,
                          plot: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Exécute la stratégie baseline complète
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame avec prix des actifs
    window : int
        Fenêtre pour le calcul du z-score
    entry_threshold : float
        Seuil d'entrée
    exit_threshold : float
        Seuil de sortie
    stop_loss : float, optional
        Stop loss en unités de z-score
    transaction_cost : float
        Coût de transaction en pourcentage
    plot : bool
        Afficher les graphiques
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        DataFrame avec résultats et dictionnaire de métriques
    """
    
    print("=" * 60)
    print("STRATÉGIE DE PAIRS TRADING BASELINE (Z-SCORE)")
    print("=" * 60)
    print(f"Paramètres:")
    print(f"  - Fenêtre: {window} jours")
    print(f"  - Seuil d'entrée: ±{entry_threshold}")
    print(f"  - Seuil de sortie: ±{exit_threshold}")
    print(f"  - Stop loss: {stop_loss if stop_loss else 'Non utilisé'}")
    print(f"  - Coûts de transaction: {transaction_cost:.2%}")
    print()
    
    # Calculs
    df = calculate_spread_and_zscore(data, window=window)
    df = generate_signals(df, entry_threshold=entry_threshold, 
                          exit_threshold=exit_threshold, 
                          stop_loss=stop_loss)
    df = calculate_returns(df, transaction_cost=transaction_cost)
    
    # Métriques
    metrics = calculate_metrics(df)
    
    # Affichage des métriques
    print("RÉSULTATS:")
    print(f"  - Période: {df.index[0].date()} à {df.index[-1].date()}")
    print(f"  - Jours de trading: {len(df)}")
    print(f"  - Rendement total (brut): {metrics['total_return_gross']:.2%}")
    print(f"  - Rendement total (net): {metrics['total_return_net']:.2%}")
    print(f"  - Rendement annualisé (net): {metrics['annualized_return_net']:.2%}")
    print(f"  - Volatilité annualisée: {metrics['annualized_vol']:.2%}")
    print(f"  - Ratio de Sharpe (net): {metrics['sharpe_ratio_net']:.2f}")
    print(f"  - Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  - Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  - Nombre de trades: {metrics['num_trades']:.0f}")
    print(f"  - Trades par an: {metrics['trades_per_year']:.1f}")
    print(f"  - Profit Factor: {metrics['profit_factor']:.2f}")
    
    if plot:
        plot_results(df, entry_threshold=entry_threshold, 
                     exit_threshold=exit_threshold)
    
    return df, metrics


if __name__ == "__main__":
    # Exemple d'utilisation
    from pathlib import Path
    
    # Charger les données
    data_path = Path("../data/pairs_data.csv")
    if data_path.exists():
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Exécuter la stratégie
        results, metrics = run_baseline_strategy(
            data,
            window=20,
            entry_threshold=2.0,
            exit_threshold=0.5,
            transaction_cost=0.001
        )
    else:
        print(f"Fichier non trouvé: {data_path}")