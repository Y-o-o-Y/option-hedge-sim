import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 下載歷史資料
stocklist=['NVDA', 'AMZN','MCD','WMT','GOOG','^IXIC']
data = yf.download(stocklist, start='2020-01-01', end='2025-03-01', interval="1d",auto_adjust=True)['Close']

for symbol in stocklist:
    print(f"\n=== {symbol} ===")
    price = data[symbol]
    log_return = np.log(price / price.shift(1))
    volatility = log_return.rolling(window=7).std().dropna()

    if volatility.empty:
        print("❌ Volatility series is empty.")
        continue

    result = adfuller(volatility)
    print(f"📊 ADF Test Statistic: {result[0]:.4f}")
    print(f"📉 p-value: {result[1]:.4f}")
    print("📐 Critical Values:")
    for level, value in result[4].items():
        print(f"{level:>3} : {value:.4f}")

    threshold = result[4]['5%']
    if result[0] < threshold:
        print("✅ 波動性具均值回歸特性")
    else:
        print("❌ 波動性無明顯均值回歸")
