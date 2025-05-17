import pandas as pd
import yfinance as yf
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# Return Data
tickers = ['NVDA']
data = yf.download(tickers, start='2020-01-05', end='2025-05-15', interval='1d',auto_adjust=True)
data = data['Close']
# Calculate the continuous returns
data_return = np.log(data / data.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()

# 建立並擬合 GARCH(1,1) 模型
model = arch_model(data_return, vol='GARCH', p=1, q=1)
res = model.fit()

################預測未來 10 天的波動（變異數）######################
forecast = res.forecast(horizon=30)
pred_var = forecast.variance.iloc[-1]
pred_vol = np.sqrt(pred_var)

vol_df = pd.DataFrame({
    "Day": np.arange(1, 31),
    "Predicted Variance": pred_var.values,
    "Predicted Volatility": pred_vol.values
})

# 只轉換數值欄位為百分比
vol_df["Predicted Variance"] *= 100
vol_df["Predicted Volatility"] *= 100

print(vol_df)

plt.figure(figsize=(10, 4))
plt.plot(vol_df["Day"], vol_df["Predicted Volatility"], marker="o")
plt.title("GARCH(1,1) Predicted 10-Day Daily Volatility")
plt.xlabel("Future Day")
plt.ylabel("Volatility (σ)")
plt.grid(True)
plt.tight_layout()
plt.show()

########################################report########################################
# # 印出參數與統計顯著性
# print(res.summary())

# # 畫圖：條件變異數（波動率平方）
# plt.figure(figsize=(12, 4))
# plt.plot(res.conditional_volatility ** 2, label='Conditional Variance')
# plt.title("GARCH(1,1) Conditional Variance for NVDA")
# plt.ylabel("Variance")
# plt.xlabel("Time")
# plt.legend()
# plt.tight_layout()
# plt.show()
