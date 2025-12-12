"""
PATH ORACLE - No Trading, Just Truth
======================================
The Gaussian already contains all paths.
We don't trade. We just show what IS.

+1 + -1 = bell curve
Position on bell = which path you're on
All paths are visible simultaneously.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import yfinance as yf


def get_all_paths(prices: np.ndarray, future_steps: int = 20) -> Dict:
    """
    Extract all paths from the Gaussian.
    No prediction. Just revelation of what's already there.
    
    Args:
        prices: Array of historical price data
        future_steps: Number of future time steps to project (default: 20)
        
    Returns:
        Dictionary containing all paths, probabilities, and current state
        
    Raises:
        ValueError: If prices array is empty or has insufficient data
        TypeError: If prices cannot be converted to float array
    """
    # Input validation
    if prices is None:
        raise ValueError("Prices array cannot be None")
    
    try:
        x = np.asarray(prices, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Prices must be convertible to float array: {e}")
    
    if len(x) == 0:
        raise ValueError("Prices array cannot be empty")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 data points for analysis")
    
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise ValueError("Prices array contains NaN or infinite values")
    
    if np.any(x <= 0):
        raise ValueError("All prices must be positive (required for log transformation)")
    
    if future_steps < 1:
        raise ValueError("future_steps must be at least 1")
    
    # Add small epsilon to prevent log(0), though we've already validated x > 0
    log_x = np.log(x + 1e-9)
    
    # Delta and Echo - calculate price changes in log space
    delta = np.diff(log_x)
    
    if len(delta) == 0:
        raise ValueError("Need at least 2 prices to calculate delta")
    
    echo = -delta[::-1]
    diff_map = delta + echo
    
    # Gaussian parameters - calculate mean and standard deviation
    mu = np.mean(diff_map)
    std = np.std(diff_map)
    
    # Ensure std is not too small to prevent division issues
    MIN_STD = 1e-9
    DEFAULT_STD = 0.01
    if std < MIN_STD:
        std = DEFAULT_STD
    
    current_price = x[-1]
    
    # Calculate current z-score (position on the bell curve)
    # Safe division since we've ensured std >= DEFAULT_STD
    current_z = (diff_map[-1] - mu) / std
    
    # All paths exist on the bell curve
    # We sample the probability density at each z-score
    Z_SCORE_MIN = -3.0
    Z_SCORE_MAX = 3.0
    Z_SCORE_SAMPLES = 100
    z_range = np.linspace(Z_SCORE_MIN, Z_SCORE_MAX, Z_SCORE_SAMPLES)
    probabilities = stats.norm.pdf(z_range)
    
    # Project paths: each z-score implies a trajectory
    paths = {}
    
    # The 5 canonical paths (quintiles of the Gaussian)
    # Each path represents a different position on the probability distribution
    path_z_scores = {
        'CRASH':    -2.0,   # Left tail (-2σ) - sharp decline
        'DECLINE':  -1.0,   # Below mean (-1σ) - gradual decline  
        'STABLE':    0.0,   # Center (0σ) - mean reversion
        'RISE':      1.0,   # Above mean (+1σ) - gradual rise
        'MOON':      2.0,   # Right tail (+2σ) - sharp rise
    }
    
    # Calculate probabilities for each path
    path_probs = [stats.norm.pdf(z) for z in path_z_scores.values()]
    prob_sum = sum(path_probs)
    
    # Ensure we don't divide by zero
    if prob_sum < 1e-10:
        prob_sum = 1.0
    
    DAMPING_FACTOR = 0.1  # Controls how aggressively paths diverge
    
    for name, z in path_z_scores.items():
        # Probability of this path (normalized across the 5 canonical paths)
        prob = stats.norm.pdf(z)
        normalized_prob = prob / prob_sum
        
        # Project price along this path
        projected = [current_price]
        for i in range(future_steps):
            # Each step moves by z * std in log space
            # The damping factor prevents unrealistic projections
            log_change = z * std * DAMPING_FACTOR
            next_price = projected[-1] * np.exp(log_change)
            
            # Sanity check: ensure projected price is reasonable
            if next_price <= 0 or np.isnan(next_price) or np.isinf(next_price):
                next_price = projected[-1]  # Keep previous price if calculation fails
            
            projected.append(next_price)
        
        # Calculate percentage change from current price
        price_change_pct = ((projected[-1] - current_price) / current_price * 100) if current_price > 0 else 0.0
        
        paths[name] = {
            'z_score': z,
            'probability': normalized_prob,
            'trajectory': np.array(projected),
            'end_price': projected[-1],
            'change_pct': price_change_pct
        }
    
    # Determine current path (where we actually are on the distribution)
    # Find the closest canonical path to our current z-score
    PROXIMITY_THRESHOLD = 0.5  # How close to a path we need to be to match it
    current_path = None
    min_distance = float('inf')
    
    for name, z in path_z_scores.items():
        distance = abs(current_z - z)
        if distance < min_distance:
            min_distance = distance
            if distance < PROXIMITY_THRESHOLD:
                current_path = name
    
    # If not close to any canonical path, classify based on z-score ranges
    if current_path is None:
        if abs(current_z) < 1.0:
            current_path = 'STABLE'
        elif current_z > 0:
            current_path = 'RISE'
        else:
            current_path = 'DECLINE'
    
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


def visualize_paths(symbol: str = "AAPL", period: str = "3mo") -> Dict:
    """
    Visualize all paths - the oracle view.
    
    Args:
        symbol: Stock ticker symbol (default: "AAPL")
        period: Time period for historical data (default: "3mo")
        
    Returns:
        Dictionary containing oracle analysis results
        
    Raises:
        ValueError: If symbol is invalid or data cannot be retrieved
        RuntimeError: If visualization fails
    """
    # Input validation
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    
    if not period or not isinstance(period, str):
        raise ValueError("Period must be a non-empty string")
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
    except Exception as e:
        raise ValueError(f"Failed to fetch data for {symbol}: {e}")
    
    if hist is None or hist.empty:
        raise ValueError(f"No historical data available for {symbol} with period {period}")
    
    if 'Close' not in hist.columns:
        raise ValueError(f"Close price data not found for {symbol}")
    
    prices = hist['Close'].values
    
    # Remove any NaN values
    prices = prices[~np.isnan(prices)]
    
    if len(prices) < 2:
        raise ValueError(f"Insufficient price data for {symbol}: need at least 2 points, got {len(prices)}")
    
    try:
        oracle = get_all_paths(prices)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"Failed to analyze paths for {symbol}: {e}")
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'PATH ORACLE: {symbol} @ ${oracle["current_price"]:.2f}', 
                     fontsize=14, fontweight='bold')
    except Exception as e:
        raise RuntimeError(f"Failed to create visualization: {e}")
    
    # 1. The Bell Curve (all paths exist here)
    ax1 = axes[0]
    try:
        ax1.fill_between(oracle['z_range'], oracle['probabilities'], alpha=0.3, color='blue')
        ax1.plot(oracle['z_range'], oracle['probabilities'], 'b-', linewidth=2)
        ax1.axvline(oracle['current_z'], color='red', linewidth=2, label=f'YOU ARE HERE (z={oracle["current_z"]:.2f})')
        
        # Mark the 5 paths with distinct colors
        colors = {'CRASH': 'darkred', 'DECLINE': 'red', 'STABLE': 'gray', 'RISE': 'green', 'MOON': 'darkgreen'}
        
        # Maximum probability for text positioning
        max_prob = np.max(oracle['probabilities']) if len(oracle['probabilities']) > 0 else 0.4
        text_y_position = max_prob * 1.05  # Place text slightly above peak
        
        for name, data in oracle['paths'].items():
            color = colors.get(name, 'gray')  # Default to gray if color not found
            ax1.axvline(data['z_score'], color=color, linestyle='--', alpha=0.5)
            ax1.text(data['z_score'], text_y_position, name, ha='center', fontsize=8, color=color)
    except Exception as e:
        print(f"Warning: Error plotting bell curve: {e}")
    
    ax1.set_xlabel('Z-Score')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('THE BELL CURVE\n(All Paths Exist)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. All Path Trajectories
    ax2 = axes[1]
    try:
        # Historical prices
        ax2.plot(range(len(prices)), prices, 'b-', linewidth=1, alpha=0.5, label='History')
        
        # Future paths - project from current time forward
        future_x = np.arange(len(prices), len(prices) + len(oracle['paths']['STABLE']['trajectory']))
        
        for name, data in oracle['paths'].items():
            # Highlight the current path more prominently
            alpha = 0.8 if name == oracle['current_path'] else 0.3
            lw = 3 if name == oracle['current_path'] else 1
            color = colors.get(name, 'gray')
            
            ax2.plot(future_x, data['trajectory'], color=color, 
                    linewidth=lw, alpha=alpha, label=f'{name} ({data["probability"]:.0%})')
        
        # Vertical line separating history from future
        ax2.axvline(len(prices) - 1, color='orange', linestyle=':', alpha=0.5)
        ax2.set_title(f'ALL PATHS\nCurrent: {oracle["current_path"]}')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
    except Exception as e:
        print(f"Warning: Error plotting trajectories: {e}")
    
    # 3. Path Probabilities
    ax3 = axes[2]
    try:
        names = list(oracle['paths'].keys())
        probs = [oracle['paths'][n]['probability'] * 100 for n in names]
        changes = [oracle['paths'][n]['change_pct'] for n in names]
        
        # Get colors for each path, with fallback
        bar_colors = [colors.get(n, 'gray') for n in names]
        bars = ax3.barh(names, probs, color=bar_colors, alpha=0.7)
        
        # Highlight current path with stronger visual emphasis
        for i, name in enumerate(names):
            if name == oracle['current_path']:
                bars[i].set_alpha(1.0)
                bars[i].set_edgecolor('white')
                bars[i].set_linewidth(2)
        
        ax3.set_xlabel('Probability %')
        ax3.set_title('PATH PROBABILITIES')
        
        # Add price targets and percentage changes
        max_prob = max(probs) if probs else 1.0
        for i, (name, change) in enumerate(zip(names, changes)):
            end_price = oracle['paths'][name]['end_price']
            # Ensure we have room for the text
            text_x = probs[i] + 1 if probs[i] < max_prob * 0.9 else probs[i] - 1
            ax3.text(text_x, i, f'${end_price:.0f} ({change:+.1f}%)', 
                    va='center', fontsize=9)
        
        # Set x-axis limits with some padding
        ax3.set_xlim(0, max_prob + 20)
        ax3.grid(True, alpha=0.3, axis='x')
    except Exception as e:
        print(f"Warning: Error plotting probabilities: {e}")
    
    try:
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Warning: Error displaying plot: {e}")
    
    # Print summary to console
    try:
        SEPARATOR_WIDTH = 60
        print(f"\n{'='*SEPARATOR_WIDTH}")
        print(f"  PATH ORACLE: {symbol}")
        print(f"{'='*SEPARATOR_WIDTH}")
        print(f"  Current Price: ${oracle['current_price']:.2f}")
        print(f"  Current Z:     {oracle['current_z']:+.2f}")
        print(f"  Current Path:  {oracle['current_path']}")
        print(f"\n  ALL PATHS:")
        
        for name, data in oracle['paths'].items():
            # Mark the current path with an arrow
            marker = ">>>" if name == oracle['current_path'] else "   "
            print(f"  {marker} {name:8} | {data['probability']:5.1%} | "
                  f"${data['end_price']:.2f} ({data['change_pct']:+.1f}%)")
        
        print(f"{'='*SEPARATOR_WIDTH}\n")
    except Exception as e:
        print(f"Warning: Error printing summary: {e}")
    
    return oracle


if __name__ == "__main__":
    # Example usage with error handling
    symbols = ["AAPL", "SPY", "TSLA"]
    
    for sym in symbols:
        try:
            print(f"\nAnalyzing {sym}...")
            visualize_paths(sym)
            input("Press Enter for next symbol...")
        except ValueError as e:
            print(f"Error analyzing {sym}: {e}")
            print("Skipping to next symbol...")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error analyzing {sym}: {e}")
            print("Skipping to next symbol...")
