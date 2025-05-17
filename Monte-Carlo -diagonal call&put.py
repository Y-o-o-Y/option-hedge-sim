import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# === Black-Scholes Pricing ===
def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# === IV Update Function ===
def update_iv(current_iv, daily_return):
    noise = np.random.normal(0, 0.01)
    if daily_return <= -0.029:
        d_iv = +0.10 + noise
    elif daily_return <= -0.01:
        d_iv = +0.02 + noise
    elif daily_return >= 0.029:
        d_iv = -0.05 + noise
    elif daily_return >= 0.01:
        d_iv = -0.02 + noise
    else:
        d_iv = noise
    return max(current_iv + d_iv, 0.01)

# === Alternating Call/Put Simulation ===
def run_alternating_diagonal_simulation(
    num_days=252,
    num_trials=1,
    initial_price=60,
    r=0.045,
    sigma_annual=0.17,
    mu_annual=0.08,
    iv_call=0.6,
    iv_put_long=0.8,
    iv_put_short=0.7,
    K_long_call_pct=1.10,
    K_short_call_pct=1.05,
    K_long_put_pct=0.90,
    K_short_put_pct=0.95,
    jump_prob=0.02,
    jump_std=0.04,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    sigma_daily = sigma_annual / np.sqrt(252)
    mu_daily = mu_annual / 252
    results = []

    for _ in range(num_trials):
        pnl_total = 0
        S = initial_price
        iv_long = iv_put_long
        iv_short = iv_put_short

        for day in range(num_days):
            sigma = sigma_daily * (1.6 if np.random.rand() < 0.02 else 0.8)
            jump = np.random.normal(0, jump_std) if np.random.rand() < jump_prob else 0
            move = mu_daily + np.random.standard_t(df=5) * sigma + jump
            S_new = S * (1 + 3 * move)
            daily_return = (S_new - S) / S

            if day % 2 == 0:  #偶數天call
                K_long = S * K_long_call_pct
                K_short = S * K_short_call_pct
                T_long, T_short = (13 - 1) / 365, (6 - 1) / 365
                long_entry = bs_price(S, K_long, 13 / 365, r, iv_call, 'call')
                short_entry = bs_price(S, K_short, 6 / 365, r, iv_call, 'call')
                long_exit = bs_price(S_new, K_long, T_long, r, iv_call, 'call')
                short_exit = bs_price(S_new, K_short, T_short, r, iv_call, 'call')
            else:   #奇數天put
                K_long = S * K_long_put_pct
                K_short = S * K_short_put_pct
                T_long, T_short = (13 - 1) / 365, (6 - 1) / 365
                iv_long = update_iv(iv_long, daily_return)
                iv_short = update_iv(iv_short, daily_return)
                long_entry = bs_price(S, K_long, 13 / 365, r, iv_long, 'put')
                short_entry = bs_price(S, K_short, 6 / 365, r, iv_short, 'put')
                long_exit = bs_price(S_new, K_long, T_long, r, iv_long, 'put')
                short_exit = bs_price(S_new, K_short, T_short, r, iv_short, 'put')

            pnl = (long_exit - long_entry - short_exit + short_entry) * 100
            pnl_total += pnl
            S = S_new

        results.append(pnl_total)

    return np.array(results)

# === Visualized Analysis ===
def analyze_diagonal_simulation(num_days=252, num_trials=500, seed=42):
    pnl_array = run_alternating_diagonal_simulation(
        num_days=num_days,
        num_trials=num_trials,
        seed=seed
    )
    mean_pnl = pnl_array.mean()
    median_pnl = np.median(pnl_array)
    std_pnl = pnl_array.std()

    sns.histplot(pnl_array, bins=50, kde=True, color='teal')
    plt.axvline(mean_pnl, color='red', linestyle='--', label=f'Mean = {mean_pnl:.2f}')
    plt.axvline(median_pnl, color='orange', linestyle='--', label=f'Median = {median_pnl:.2f}')
    plt.title(f"PnL Distribution: Alternating Diagonal Strategy\n({num_trials} Trials, {num_days} Days Each)")
    plt.xlabel("Total PnL ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        "PnL": pnl_array,
        "Mean": [mean_pnl] * num_trials,
        "Median": [median_pnl] * num_trials,
        "Std": [std_pnl] * num_trials
    })
analyze_diagonal_simulation(num_days=120, num_trials=500, seed=42)