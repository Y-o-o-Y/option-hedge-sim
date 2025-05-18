import yfinance as yf
from datetime import datetime
import pandas as pd

def find_nearest_expiry(desired_days, ticker="NVDA", tolerance=3):
    t = yf.Ticker(ticker)
    today = datetime.today()
    closest_diff = float("inf")
    best_match = None

    for exp_str in t.options:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        days_to_expiry = (exp_date - today).days
        diff = abs(desired_days - days_to_expiry)
        if diff <= tolerance and diff < closest_diff:
            best_match = (exp_str, days_to_expiry)
            closest_diff = diff

    return best_match  # 回傳 (到期日字串, 實際天數)

def get_atm_iv(desired_days, ticker="NVDA"):
    expiry_data = find_nearest_expiry(desired_days, ticker)
    if not expiry_data:
        return None, None, None

    expiry_str, actual_days = expiry_data
    t = yf.Ticker(ticker)
    spot = t.history(period="1d")["Close"][-1]
    calls = t.option_chain(expiry_str).calls.copy()
    calls["abs_diff"] = abs(calls["strike"] - spot)
    atm_call = calls.sort_values("abs_diff").iloc[0]
    atm_iv = atm_call["impliedVolatility"]

    return atm_iv, expiry_str, actual_days

# 使用範例（可刪除）
if __name__ == "__main__":
    iv, expiry, days = get_atm_iv(10)
    if iv:
        print(f"你輸入10天 → 使用 {expiry}（實際{days}天） → ATM IV: {iv:.2%}")
    else:
        print("⚠️ 找不到合適的期權到期日。")
