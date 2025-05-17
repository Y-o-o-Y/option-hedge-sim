import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Input the tickers
stock3 = '^VIX'

tickers = ['NVDA', 'AMZN','TSLA','WMT','^IXIC',stock3]
data = yf.download(tickers, start='2020-01-05', end='2025-03-16', interval='1d',auto_adjust=True)
data = data['Close']
# Calculate the continuous returns
data_return = np.log(data/data.shift(1))
#####################

STDEV_of_return = data_return.std()*100

avg_ln_return = data_return.mean() * 100

correlation_with_stock3 = data_return.corr()[stock3].drop(stock3)
drift_to_stdev_ratio= (avg_ln_return*252 / (STDEV_of_return*np.sqrt(252))).round(2) #Perferr higher
# === Output ===
print("=== Daily Std Dev (Volatility) ===")
print(STDEV_of_return.round(2).astype(str) + '%')
# print("=== Daily log return (Volatility) ===") it's almost 0
# print(avg_ln_return.round(2).astype(str) + '%')
print("\n=== Average Annual Log Return (5Y) ===")
print(f"{(avg_ln_return * 252).round(2).astype(str)}%")
print(f"\n=== Correlation with {stock3} ===")
print(correlation_with_stock3.round(2))
print('\n===（飄移 / 波動）===') #近似衡量“走出你預期方向”的穩定性 同飄移條件下 這個比例越高越好
print(drift_to_stdev_ratio)





