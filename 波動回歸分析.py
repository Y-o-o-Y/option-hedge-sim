import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Step 2: Conditional Expectation (Binned)
def conditional_mean_reversion(vol_series, bins=[-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]):
    df = pd.DataFrame({'vol': vol_series})
    df['vol_lag'] = df['vol'].shift(1)
    df.dropna(inplace=True)
    df['bin'] = pd.cut(df['vol_lag'], bins=bins)
    grouped = df.groupby('bin')['vol'].mean()
    
    plt.figure(figsize=(8,5))
    grouped.plot(kind='bar', color='skyblue')
    plt.axhline(vol_series.mean(), color='red', linestyle='--', label='Mean Volatility')
    plt.title('Conditional Mean Reversion: Next-Day Volatility by Lag Bin')
    plt.ylabel('Next-Day Volatility')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return grouped

# Step 3: OU Model Fitting  Ornstein-Uhlenbeck Process OU 本身是對稱的
def fit_ou_model(vol_series):
    df = pd.DataFrame({'vol': vol_series})
    df['vol_lag'] = df['vol'].shift(1)
    df.dropna(inplace=True)

    X = df[['vol_lag']]
    y = df['vol']
    reg = LinearRegression().fit(X, y)
    theta = 1 - reg.coef_[0]
    mu = reg.intercept_ / theta if theta != 0 else np.nan
    sigma = np.std(y - reg.predict(X))

    print(f"\n📘 OU 模型擬合結果 (以 AR(1) 為近似)：")
    print(f"   回歸力 θ: {theta:.4f}")
    print(f"   長期均值 μ: {mu:.4f}")
    print(f"   擾動波幅 σ: {sigma:.4f}")

# Step 4: Asymmetric Regression
def asymmetric_regression(vol_series):
    df = pd.DataFrame({'vol': vol_series})
    df['vol_lag'] = df['vol'].shift(1)
    df.dropna(inplace=True)

    df['pos'] = df['vol_lag'].apply(lambda x: x if x >= 0 else 0)
    df['neg'] = df['vol_lag'].apply(lambda x: x if x < 0 else 0)
    
    X = df[['pos', 'neg']]
    X = sm.add_constant(X)
    y = df['vol']
    
    model = sm.OLS(y, X).fit()
    print("\n📊 非對稱均值回歸結果（OLS）：")
    print(model.summary())


if __name__ == "__main__":
# === Main: Download data & run analysis ===
    data = yf.download("NVDA", start="2020-01-01", end="2025-03-01", interval="1d")
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['volatility'] = data['log_return'] ** 2
    vol_series = data['volatility'].dropna()

# Run analysis
    conditional_mean_reversion(vol_series)
    fit_ou_model(vol_series)
    asymmetric_regression(vol_series)
