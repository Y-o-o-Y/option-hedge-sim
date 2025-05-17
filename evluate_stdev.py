import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

tickers = ['NVDA','WMT']
data = yf.download(tickers, start='2020-01-01', end='2025-03-01', interval='1d',auto_adjust=True)
data = data['Close']
# Calculate the continuous returns
data_return = np.log(data/data.shift(1))
#####################
STDEV_of_return = data_return.std()

prob_exceed_1std = (data_return.abs() > STDEV_of_return).sum() / data_return.count()

# 顯示結果  理論上是32%
for ticker in tickers:
    print(f"{ticker} 絕對報酬超過 1 標準差的機率: {prob_exceed_1std[ticker]:.2%}")


