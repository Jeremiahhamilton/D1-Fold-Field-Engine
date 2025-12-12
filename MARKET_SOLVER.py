"""
PATH ORACLE - No Trading, Just Truth
======================================
The Gaussian already contains all paths.
We don't trade. We just show what IS.

+1 + -1 = bell curve
Position on bell = which path you're on
All paths are visible simultaneously.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import yfinance as yf


def get_all_paths(prices: np.ndarray, future_steps: int = 20) -> dict:
    """
    Extract all paths from the Gaussian.
    No prediction. Just revelation of what's already there.
    """
    x = np.asarray(prices, dtype=float)
    log_x = np.log(x + 1e-9)
    
    # Delta and Echo
    delta = np.diff(log_x)
    echo = -delta[::-1]
    diff_map = delta + echo
    
    # Gaussian parameters
    mu, std = np.mean(diff_map), np.std(diff_map)
    if std < 1e-9:
        std = 0.01
    
    current_price = x[-1]
    current_z = (diff_map[-1] - mu) / std
    
    # All paths exist on the bell curve
    # We sample the probability density at each z-score
    z_range = np.linspace(-3, 3, 100)
    probabilities = stats.norm.pdf(z_range)
    
    # Project paths: each z-score implies a trajectory
    paths = {}
    
    # The 5 canonical paths (quintiles of the Gaussian)
    path_z_scores = {
        'CRASH':    -2.0,   # Left tail - sharp decline
        'DECLINE':  -1.0,   # Below mean - gradual decline  
        'STABLE':    0.0,   # Center - mean reversion
        'RISE':      1.0,   # Above mean - gradual rise
        'MOON':      2.0,   # Right tail - sharp rise
    }
    
    for name, z in path_z_scores.items():
        # Probability of this path
        prob = stats.norm.pdf(z)
        
        # Project price along this path
        projected = [current_price]
        for i in range(future_steps):
            # Each step moves by z * std in log space
            log_change = z * std * 0.1  # Damped
            next_price = projected[-1] * np.exp(log_change)
            projected.append(next_price)
        
        paths[name] = {
            'z_score': z,
            'probability': prob / sum(stats.norm.pdf(list(path_z_scores.values()))),  # Normalized
            'trajectory': np.array(projected),
            'end_price': projected[-1],
            'change_pct': (projected[-1] - current_price) / current_price * 100
        }
    
    # Current path (where we actually are)
    current_path = None
    for name, z in path_z_scores.items():
        if abs(current_z - z) < 0.5:
            current_path = name
            break
    if current_path is None:
        current_path = 'STABLE' if abs(current_z) < 1 else ('RISE' if current_z > 0 else 'DECLINE')
    
    return {
        'current_price': current_price,
        'current_z': current_z,
        'current_path': current_path,
        'paths': paths,
        'mu': mu,
        'std': std,
        'z_range': z_range,
        'probabilities': probabilities
    }


def visualize_paths(symbol: str = "AAPL", period: str = "3mo"):
    """Visualize all paths - the oracle view."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    prices = hist['Close'].values
    
    oracle = get_all_paths(prices)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'PATH ORACLE: {symbol} @ ${oracle["current_price"]:.2f}', 
                 fontsize=14, fontweight='bold')
    
    # 1. The Bell Curve (all paths exist here)
    ax1 = axes[0]
    ax1.fill_between(oracle['z_range'], oracle['probabilities'], alpha=0.3, color='blue')
    ax1.plot(oracle['z_range'], oracle['probabilities'], 'b-', linewidth=2)
    ax1.axvline(oracle['current_z'], color='red', linewidth=2, label=f'YOU ARE HERE (z={oracle["current_z"]:.2f})')
    
    # Mark the 5 paths
    colors = {'CRASH': 'darkred', 'DECLINE': 'red', 'STABLE': 'gray', 'RISE': 'green', 'MOON': 'darkgreen'}
    for name, data in oracle['paths'].items():
        ax1.axvline(data['z_score'], color=colors[name], linestyle='--', alpha=0.5)
        ax1.text(data['z_score'], 0.42, name, ha='center', fontsize=8, color=colors[name])
    
    ax1.set_xlabel('Z-Score')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('THE BELL CURVE\n(All Paths Exist)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. All Path Trajectories
    ax2 = axes[1]
    
    # Historical
    ax2.plot(range(len(prices)), prices, 'b-', linewidth=1, alpha=0.5, label='History')
    
    # Future paths
    future_x = np.arange(len(prices), len(prices) + 21)
    for name, data in oracle['paths'].items():
        alpha = 0.8 if name == oracle['current_path'] else 0.3
        lw = 3 if name == oracle['current_path'] else 1
        ax2.plot(future_x, data['trajectory'], color=colors[name], 
                linewidth=lw, alpha=alpha, label=f'{name} ({data["probability"]:.0%})')
    
    ax2.axvline(len(prices) - 1, color='orange', linestyle=':', alpha=0.5)
    ax2.set_title(f'ALL PATHS\nCurrent: {oracle["current_path"]}')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Path Probabilities
    ax3 = axes[2]
    names = list(oracle['paths'].keys())
    probs = [oracle['paths'][n]['probability'] * 100 for n in names]
    changes = [oracle['paths'][n]['change_pct'] for n in names]
    
    bars = ax3.barh(names, probs, color=[colors[n] for n in names], alpha=0.7)
    
    # Highlight current path
    for i, name in enumerate(names):
        if name == oracle['current_path']:
            bars[i].set_alpha(1.0)
            bars[i].set_edgecolor('white')
            bars[i].set_linewidth(2)
    
    ax3.set_xlabel('Probability %')
    ax3.set_title('PATH PROBABILITIES')
    
    # Add price targets
    for i, (name, change) in enumerate(zip(names, changes)):
        end_price = oracle['paths'][name]['end_price']
        ax3.text(probs[i] + 1, i, f'${end_price:.0f} ({change:+.1f}%)', 
                va='center', fontsize=9)
    
    ax3.set_xlim(0, max(probs) + 20)
    ax3.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  PATH ORACLE: {symbol}")
    print(f"{'='*60}")
    print(f"  Current Price: ${oracle['current_price']:.2f}")
    print(f"  Current Z:     {oracle['current_z']:+.2f}")
    print(f"  Current Path:  {oracle['current_path']}")
    print(f"\n  ALL PATHS:")
    for name, data in oracle['paths'].items():
        marker = ">>>" if name == oracle['current_path'] else "   "
        print(f"  {marker} {name:8} | {data['probability']:5.1%} | ${data['end_price']:.2f} ({data['change_pct']:+.1f}%)")
    print(f"{'='*60}\n")
    
    return oracle


if __name__ == "__main__":
    for sym in ["AAPL", "SPY", "TSLA"]:
        visualize_paths(sym)
        input("Press Enter for next symbol...")
