#!/usr/bin/env python3
"""
MARKET ANALYZER - Data-Focused Multi-Market Analysis

Drop multiple CSVs, get clean data analysis:
- Individual market states
- Cross-market correlations
- Phase alignment scores
- Flip projections
- Support/Resistance levels
- Divergence detection

Less graphics, more numbers. Built for brokers.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

try:
    import tkinterdnd2 as tkdnd
    HAS_DND = True
except ImportError:
    HAS_DND = False


# ============== CORE SOLVER ============== #

def load_market_csv(path):
    """Auto-detect price column."""
    df = pd.read_csv(path)
    
    for col in ["Close", "Adj Close", "close", "adj_close", "Price", "price", 
                "SP500", "DJIA", "NASDAQ", "Value", "value", "Last", "last"]:
        if col in df.columns:
            prices = df[col].dropna().astype(float).values
            
            dates = None
            if "Date" in df.columns:
                dates = pd.to_datetime(df["Date"]).values[:len(prices)]
            elif "observation_date" in df.columns:
                dates = pd.to_datetime(df["observation_date"]).values[:len(prices)]
            
            return prices, col, dates
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        col = num_cols[0]
        prices = df[col].dropna().astype(float).values
        return prices, col, None
    
    raise ValueError("No price data found")


def project_futures(prices, regime, delta, avg_interval, periods=10):
    """
    Project future states from past geometry.
    Future is not prediction - it's the echo of the past already solved.
    """
    x = np.asarray(prices, dtype=float)
    current_price = x[-1]
    current_regime = regime[-1] if len(regime) > 0 else 0
    
    # The echo IS the future - mirrored from the past
    # Take recent delta pattern and project it forward
    recent_delta = delta[-min(20, len(delta)):]
    echo_future = -recent_delta[::-1]  # Future = inverted mirror of past
    
    # Project price levels
    projected_prices = [current_price]
    projected_regimes = [current_regime]
    
    for i in range(min(periods, len(echo_future))):
        # Price moves by echo magnitude (scaled back from log space)
        price_change = echo_future[i] * current_price * 0.1  # Scale factor
        next_price = projected_prices[-1] + price_change
        projected_prices.append(next_price)
        
        # Regime flips based on avg interval
        if (i + 1) % max(1, int(avg_interval)) == 0:
            projected_regimes.append(-projected_regimes[-1] if projected_regimes[-1] != 0 else 1)
        else:
            projected_regimes.append(projected_regimes[-1])
    
    # Key future levels
    future_high = max(projected_prices)
    future_low = min(projected_prices)
    
    # Next flip timing
    flips_in_projection = sum(1 for i in range(1, len(projected_regimes)) 
                              if projected_regimes[i] != projected_regimes[i-1])
    
    return {
        "projected_prices": projected_prices[1:],  # Exclude current
        "projected_regimes": projected_regimes[1:],
        "future_high": future_high,
        "future_low": future_low,
        "future_range": future_high - future_low,
        "flips_in_projection": flips_in_projection,
        "projection_periods": len(projected_prices) - 1,
        "end_regime": "BULLISH" if projected_regimes[-1] > 0 else "BEARISH" if projected_regimes[-1] < 0 else "NEUTRAL",
    }


def analyze_market(prices):
    """Full market analysis - returns comprehensive data."""
    x = np.asarray(prices, dtype=float)
    n = len(x)
    
    if n < 3:
        raise ValueError("Need at least 3 data points")
    
    # Basic stats
    current_price = x[-1]
    high = np.max(x)
    low = np.min(x)
    mean = np.mean(x)
    std = np.std(x)
    
    # Returns
    returns = np.diff(x) / x[:-1] * 100
    total_return = (x[-1] - x[0]) / x[0] * 100
    avg_daily_return = np.mean(returns)
    volatility = np.std(returns)
    
    # Delta fold geometry
    log_x = np.log(x + 1e-9)
    log_centered = log_x - np.mean(log_x)
    delta = np.diff(log_centered)
    m = len(delta)
    
    echo = -delta[::-1]
    mag = np.hypot(delta, echo)
    phase = np.arctan2(echo, delta)
    mag_norm = mag / (np.max(mag) + 1e-9)
    regime = np.sign(delta * echo)
    
    # Flip analysis
    flips = np.where(np.diff(regime) != 0)[0] + 1
    current_regime = regime[-1] if m > 0 else 0
    
    if len(flips) >= 2:
        intervals = np.diff(flips)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        last_flip = flips[-1]
        since_last = m - 1 - last_flip
        to_next = max(0, avg_interval - since_last)
        confidence = 1.0 / (1.0 + std_interval / (avg_interval + 1e-9))
    else:
        avg_interval = m
        std_interval = 0
        to_next = m
        confidence = 0.5
    
    # Support/Resistance (vectorized to avoid repeated Python slicing work)
    support = []
    resistance = []
    if m > 20:
        idx = np.arange(10, m - 10)
        prev_windows = sliding_window_view(mag_norm, 10)
        next_windows = sliding_window_view(mag_norm[1:], 10)

        prev_max = np.max(prev_windows[idx - 10], axis=1)
        next_max = np.max(next_windows[idx - 10], axis=1)
        prev_min = np.min(prev_windows[idx - 10], axis=1)
        next_min = np.min(next_windows[idx - 10], axis=1)

        local_vals = mag_norm[idx]
        resistance_mask = (local_vals > prev_max) & (local_vals > next_max)
        support_mask = (local_vals < prev_min) & (local_vals < next_min)

        resistance = x[idx + 1][resistance_mask].tolist()
        support = x[idx + 1][support_mask].tolist()
    
    # Trend strength (magnitude of recent moves)
    recent_mag = np.mean(mag_norm[-20:]) if len(mag_norm) >= 20 else np.mean(mag_norm)
    trend_strength = "STRONG" if recent_mag > 0.5 else "MODERATE" if recent_mag > 0.25 else "WEAK"
    
    # Phase momentum
    recent_phase = phase[-10:] if len(phase) >= 10 else phase
    phase_momentum = np.mean(np.diff(recent_phase)) if len(recent_phase) > 1 else 0
    
    return {
        # Basic
        "data_points": n,
        "current_price": current_price,
        "high": high,
        "low": low,
        "mean": mean,
        "std": std,
        
        # Returns
        "total_return_pct": total_return,
        "avg_daily_return_pct": avg_daily_return,
        "volatility_pct": volatility,
        
        # Regime
        "current_regime": current_regime,
        "current_trend": "BULLISH" if current_regime > 0 else "BEARISH" if current_regime < 0 else "NEUTRAL",
        "trend_strength": trend_strength,
        "confidence": confidence,
        
        # Flips
        "total_flips": len(flips),
        "avg_flip_interval": avg_interval,
        "std_flip_interval": std_interval,
        "distance_to_next_flip": to_next,
        "next_regime": "BEARISH" if current_regime > 0 else "BULLISH" if current_regime < 0 else "UNKNOWN",
        
        # Levels
        "support_levels": sorted(set(support))[-5:] if support else [],
        "resistance_levels": sorted(set(resistance))[:5] if resistance else [],
        
        # Phase
        "phase_momentum": phase_momentum,
        "recent_intensity": recent_mag,
        
        # Raw data for correlation
        "regime": regime,
        "phase": phase,
        "mag_norm": mag_norm,
        "delta": delta,
    }
    
    # Add futures projection
    futures = project_futures(x, regime, delta, avg_interval, periods=10)
    result["futures"] = futures
    
    return result


def analyze_market(prices):
    """Full market analysis - returns comprehensive data."""
    x = np.asarray(prices, dtype=float)
    n = len(x)
    
    if n < 3:
        raise ValueError("Need at least 3 data points")
    
    # Basic stats
    current_price = x[-1]
    high = np.max(x)
    low = np.min(x)
    mean = np.mean(x)
    std = np.std(x)
    
    # Returns
    returns = np.diff(x) / x[:-1] * 100
    total_return = (x[-1] - x[0]) / x[0] * 100
    avg_daily_return = np.mean(returns)
    volatility = np.std(returns)
    
    # Delta fold geometry
    log_x = np.log(x + 1e-9)
    log_centered = log_x - np.mean(log_x)
    delta = np.diff(log_centered)
    m = len(delta)
    
    echo = -delta[::-1]
    mag = np.hypot(delta, echo)
    phase = np.arctan2(echo, delta)
    mag_norm = mag / (np.max(mag) + 1e-9)
    regime = np.sign(delta * echo)
    
    # Flip analysis
    flips = np.where(np.diff(regime) != 0)[0] + 1
    current_regime = regime[-1] if m > 0 else 0
    
    if len(flips) >= 2:
        intervals = np.diff(flips)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        last_flip = flips[-1]
        since_last = m - 1 - last_flip
        to_next = max(0, avg_interval - since_last)
        confidence = 1.0 / (1.0 + std_interval / (avg_interval + 1e-9))
    else:
        avg_interval = m
        std_interval = 0
        to_next = m
        confidence = 0.5
    
    # Support/Resistance (vectorized to avoid repeated Python slicing work)
    support = []
    resistance = []
    if m > 20:
        idx = np.arange(10, m - 10)
        prev_windows = sliding_window_view(mag_norm, 10)
        next_windows = sliding_window_view(mag_norm[1:], 10)

        prev_max = np.max(prev_windows[idx - 10], axis=1)
        next_max = np.max(next_windows[idx - 10], axis=1)
        prev_min = np.min(prev_windows[idx - 10], axis=1)
        next_min = np.min(next_windows[idx - 10], axis=1)

        local_vals = mag_norm[idx]
        resistance_mask = (local_vals > prev_max) & (local_vals > next_max)
        support_mask = (local_vals < prev_min) & (local_vals < next_min)

        resistance = x[idx + 1][resistance_mask].tolist()
        support = x[idx + 1][support_mask].tolist()
    
    # Trend strength
    recent_mag = np.mean(mag_norm[-20:]) if len(mag_norm) >= 20 else np.mean(mag_norm)
    trend_strength = "STRONG" if recent_mag > 0.5 else "MODERATE" if recent_mag > 0.25 else "WEAK"
    
    # Phase momentum
    recent_phase = phase[-10:] if len(phase) >= 10 else phase
    phase_momentum = np.mean(np.diff(recent_phase)) if len(recent_phase) > 1 else 0
    
    result = {
        "data_points": n,
        "current_price": current_price,
        "high": high,
        "low": low,
        "mean": mean,
        "std": std,
        "total_return_pct": total_return,
        "avg_daily_return_pct": avg_daily_return,
        "volatility_pct": volatility,
        "current_regime": current_regime,
        "current_trend": "BULLISH" if current_regime > 0 else "BEARISH" if current_regime < 0 else "NEUTRAL",
        "trend_strength": trend_strength,
        "confidence": confidence,
        "total_flips": len(flips),
        "avg_flip_interval": avg_interval,
        "std_flip_interval": std_interval,
        "distance_to_next_flip": to_next,
        "next_regime": "BEARISH" if current_regime > 0 else "BULLISH" if current_regime < 0 else "UNKNOWN",
        "support_levels": sorted(set(support))[-5:] if support else [],
        "resistance_levels": sorted(set(resistance))[:5] if resistance else [],
        "phase_momentum": phase_momentum,
        "recent_intensity": recent_mag,
        "regime": regime,
        "phase": phase,
        "mag_norm": mag_norm,
        "delta": delta,
    }
    
    # Add futures projection
    futures = project_futures(x, regime, delta, avg_interval, periods=10)
    result["futures"] = futures
    
    return result


def cross_market_analysis(markets):
    """Analyze relationships between multiple markets."""
    names = list(markets.keys())
    n = len(names)
    
    if n < 2:
        return None
    
    results = {
        "market_count": n,
        "markets": names,
        "correlations": {},
        "alignments": {},
        "system_state": "",
        "divergences": [],
    }
    
    # Pairwise analysis
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i >= j:
                continue
            
            pair = f"{name_i}/{name_j}"
            m_i = markets[name_i]
            m_j = markets[name_j]
            
            # Align lengths
            min_len = min(len(m_i["regime"]), len(m_j["regime"]))
            
            if min_len > 0:
                r_i = m_i["regime"][-min_len:]
                r_j = m_j["regime"][-min_len:]
                
                # Regime alignment
                matches = np.sum(r_i == r_j) / min_len
                
                # Phase correlation
                p_i = m_i["phase"][-min_len:]
                p_j = m_j["phase"][-min_len:]
                phase_corr = np.corrcoef(p_i, p_j)[0, 1]
                
                # Current alignment
                curr_i = m_i["current_regime"]
                curr_j = m_j["current_regime"]
                
                if curr_i == curr_j and curr_i != 0:
                    current_align = "IN-PHASE"
                elif curr_i == -curr_j and curr_i != 0:
                    current_align = "ANTI-PHASE"
                else:
                    current_align = "MIXED"
                
                results["correlations"][pair] = {
                    "regime_alignment": matches,
                    "phase_correlation": phase_corr if not np.isnan(phase_corr) else 0,
                    "current_alignment": current_align,
                }
                
                # Check for divergence
                if current_align == "ANTI-PHASE":
                    results["divergences"].append(f"{name_i} ({m_i['current_trend']}) vs {name_j} ({m_j['current_trend']})")
    
    # Overall system state
    regimes = [markets[name]["current_regime"] for name in names]
    bulls = sum(1 for r in regimes if r > 0)
    bears = sum(1 for r in regimes if r < 0)
    neutrals = sum(1 for r in regimes if r == 0)
    
    if bulls == n:
        results["system_state"] = "FULL BULL ALIGNMENT"
    elif bears == n:
        results["system_state"] = "FULL BEAR ALIGNMENT"
    elif neutrals == n:
        results["system_state"] = "ALL NEUTRAL"
    elif bulls > bears:
        results["system_state"] = f"BULL MAJORITY ({bulls}/{n})"
    elif bears > bulls:
        results["system_state"] = f"BEAR MAJORITY ({bears}/{n})"
    else:
        results["system_state"] = f"SPLIT ({bulls} bull, {bears} bear)"
    
    results["bull_count"] = bulls
    results["bear_count"] = bears
    results["neutral_count"] = neutrals
    
    return results


# ============== GUI ============== #

class MarketAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MARKET ANALYZER - Data Analysis")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")
        
        self.markets = {}
        
        self._build_ui()
        self._setup_dnd()
    
    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1a1a2e")
        style.configure("TLabel", background="#1a1a2e", foreground="#eee", font=("Arial", 11))
        style.configure("Title.TLabel", font=("Arial", 18, "bold"), foreground="#00d4ff")
        style.configure("TButton", font=("Arial", 11, "bold"))
        
        main = ttk.Frame(self.root, padding=15)
        main.pack(fill="both", expand=True)
        
        # Header
        ttk.Label(main, text="MARKET ANALYZER", style="Title.TLabel").pack(anchor="w")
        ttk.Label(main, text="Drop CSVs for data analysis", style="TLabel").pack(anchor="w", pady=(0, 10))
        
        # Drop zone
        self.drop_label = tk.Label(
            main, text="📁 DROP CSV FILES HERE", font=("Arial", 12, "bold"),
            bg="#2a2a4e", fg="#00d4ff", relief="ridge", bd=2, pady=10
        )
        self.drop_label.pack(fill="x", pady=5)
        self.drop_label.bind("<Button-1>", lambda e: self.browse_files())
        
        # Controls
        ctrl = ttk.Frame(main)
        ctrl.pack(fill="x", pady=5)
        
        self.markets_var = tk.StringVar(value="No markets loaded")
        ttk.Label(ctrl, textvariable=self.markets_var, style="TLabel").pack(side="left")
        ttk.Button(ctrl, text="Clear", command=self.clear_all).pack(side="right")
        ttk.Button(ctrl, text="Analyze", command=self.run_analysis).pack(side="right", padx=5)
        ttk.Button(ctrl, text="Export", command=self.export_report).pack(side="right", padx=5)
        
        # Output
        self.output = tk.Text(
            main, wrap="word", font=("Consolas", 10),
            bg="#0d0d1a", fg="#00ff88", insertbackground="#00ff88"
        )
        self.output.pack(fill="both", expand=True, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.output, command=self.output.yview)
        scrollbar.pack(side="right", fill="y")
        self.output.config(yscrollcommand=scrollbar.set)
    
    def _setup_dnd(self):
        if HAS_DND:
            self.drop_label.drop_target_register("DND_Files")
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop)
    
    def _on_drop(self, event):
        paths = event.data.strip().split()
        for path in paths:
            path = path.strip("{}")
            if path.lower().endswith(".csv"):
                self.add_market(path)
        self.run_analysis()
    
    def browse_files(self):
        paths = filedialog.askopenfilenames(
            title="Select Market CSVs",
            filetypes=[("CSV files", "*.csv")]
        )
        for path in paths:
            self.add_market(path)
        if paths:
            self.run_analysis()
    
    def add_market(self, path):
        name = Path(path).stem
        try:
            prices, col, dates = load_market_csv(path)
            analysis = analyze_market(prices)
            analysis["file"] = Path(path).name
            analysis["col"] = col
            self.markets[name] = analysis
            self._update_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {name} - {e}")
    
    def _update_list(self):
        if self.markets:
            self.markets_var.set(f"Loaded: {', '.join(self.markets.keys())}")
        else:
            self.markets_var.set("No markets loaded")
    
    def clear_all(self):
        self.markets = {}
        self._update_list()
        self.output.delete("1.0", "end")
    
    def run_analysis(self):
        if not self.markets:
            return
        
        self.output.delete("1.0", "end")
        
        report = self._generate_report()
        self.output.insert("1.0", report)
    
    def _generate_report(self):
        lines = []
        lines.append("=" * 80)
        lines.append("                    MARKET ANALYZER - DATA REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Individual market analysis
        for name, m in self.markets.items():
            lines.append("-" * 80)
            lines.append(f"  {name.upper()}")
            lines.append("-" * 80)
            lines.append("")
            lines.append(f"  File:              {m['file']}")
            lines.append(f"  Column:            {m['col']}")
            lines.append(f"  Data Points:       {m['data_points']}")
            lines.append("")
            lines.append("  PRICE DATA")
            lines.append(f"    Current:         {m['current_price']:.2f}")
            lines.append(f"    High:            {m['high']:.2f}")
            lines.append(f"    Low:             {m['low']:.2f}")
            lines.append(f"    Mean:            {m['mean']:.2f}")
            lines.append(f"    Std Dev:         {m['std']:.2f}")
            lines.append("")
            lines.append("  RETURNS")
            lines.append(f"    Total Return:    {m['total_return_pct']:+.2f}%")
            lines.append(f"    Avg Daily:       {m['avg_daily_return_pct']:+.4f}%")
            lines.append(f"    Volatility:      {m['volatility_pct']:.4f}%")
            lines.append("")
            lines.append("  REGIME STATE")
            lines.append(f"    Current Trend:   {m['current_trend']}")
            lines.append(f"    Regime:          {'+1' if m['current_regime'] > 0 else '-1' if m['current_regime'] < 0 else '0'}")
            lines.append(f"    Strength:        {m['trend_strength']}")
            lines.append(f"    Confidence:      {m['confidence']:.1%}")
            lines.append("")
            lines.append("  FLIP ANALYSIS")
            lines.append(f"    Total Flips:     {m['total_flips']}")
            lines.append(f"    Avg Interval:    {m['avg_flip_interval']:.1f} periods")
            lines.append(f"    Std Interval:    {m['std_flip_interval']:.1f} periods")
            lines.append(f"    To Next Flip:    ~{m['distance_to_next_flip']:.0f} periods")
            lines.append(f"    Next Regime:     {m['next_regime']}")
            lines.append("")
            lines.append("  KEY LEVELS")
            lines.append(f"    Support:         {', '.join(f'{x:.2f}' for x in m['support_levels']) or 'None'}")
            lines.append(f"    Resistance:      {', '.join(f'{x:.2f}' for x in m['resistance_levels']) or 'None'}")
            lines.append("")
            
            # FUTURES - the echo of the past
            if "futures" in m:
                f = m["futures"]
                lines.append("  FUTURES (Echo Projection)")
                lines.append(f"    Periods:         {f['projection_periods']}")
                lines.append(f"    Future High:     {f['future_high']:.2f}")
                lines.append(f"    Future Low:      {f['future_low']:.2f}")
                lines.append(f"    Future Range:    {f['future_range']:.2f}")
                lines.append(f"    Flips Expected:  {f['flips_in_projection']}")
                lines.append(f"    End Regime:      {f['end_regime']}")
                if f['projected_prices']:
                    lines.append(f"    Price Path:      {', '.join(f'{p:.2f}' for p in f['projected_prices'][:5])}...")
                lines.append("")
        
        # Cross-market analysis
        if len(self.markets) >= 2:
            cross = cross_market_analysis(self.markets)
            
            lines.append("=" * 80)
            lines.append("                    CROSS-MARKET ANALYSIS")
            lines.append("=" * 80)
            lines.append("")
            lines.append(f"  SYSTEM STATE:      {cross['system_state']}")
            lines.append(f"  Bull Markets:      {cross['bull_count']}")
            lines.append(f"  Bear Markets:      {cross['bear_count']}")
            lines.append(f"  Neutral:           {cross['neutral_count']}")
            lines.append("")
            
            if cross["divergences"]:
                lines.append("  DIVERGENCES DETECTED:")
                for div in cross["divergences"]:
                    lines.append(f"    • {div}")
                lines.append("")
            
            lines.append("  PAIR CORRELATIONS")
            lines.append("  " + "-" * 60)
            lines.append(f"  {'Pair':<20} {'Regime Align':<15} {'Phase Corr':<15} {'Current':<15}")
            lines.append("  " + "-" * 60)
            
            for pair, data in cross["correlations"].items():
                lines.append(f"  {pair:<20} {data['regime_alignment']:.1%}          {data['phase_correlation']:+.3f}          {data['current_alignment']:<15}")
            
            lines.append("")
        
        # Summary
        lines.append("=" * 80)
        lines.append("                         SUMMARY")
        lines.append("=" * 80)
        lines.append("")
        
        for name, m in self.markets.items():
            trend_arrow = "↑" if m['current_regime'] > 0 else "↓" if m['current_regime'] < 0 else "→"
            lines.append(f"  {name:<15} {trend_arrow} {m['current_trend']:<10} | Next flip: ~{m['distance_to_next_flip']:.0f} → {m['next_regime']}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def export_report(self):
        if not self.markets:
            messagebox.showwarning("No Data", "Load markets first")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if path:
            report = self._generate_report()
            with open(path, "w", encoding="utf-8") as f:
                f.write(report)
            messagebox.showinfo("Exported", f"Report saved to {path}")


def main():
    if HAS_DND:
        root = tkdnd.TkinterDnD.Tk()
    else:
        root = tk.Tk()
    
    app = MarketAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
