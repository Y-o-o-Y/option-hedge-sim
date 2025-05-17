# 工具：Greeks 計算器
import math
from scipy.stats import norm

def compute_greeks(S, K, T, r, sigma, option_type='put', direction='short'):
    if T <= 0:
        return 0, 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100

    if direction == 'short':
        delta *= -1
        vega *= -1

    return round(delta, 4), round(vega, 4)

# 範例數據（可替換）
examples = [
    {"symbol": "TQQQ", "S": 54, "K": 48, "T": 32/365, "r": 0.045, "sigma": 0.84, "type": "put", "dir": "short"},
    {"symbol": "SQQQ", "S": 32, "K": 28, "T": 32/365, "r": 0.045, "sigma": 0.71, "type": "put", "dir": "short"},
]

print("Greeks 測試結果：")
total_delta = 0
for e in examples:
    delta, vega = compute_greeks(e['S'], e['K'], e['T'], e['r'], e['sigma'], e['type'], e['dir'])
    print(f"{e['symbol']}: {e['dir'].upper()} {e['type'].upper()} K={e['K']} Delta={delta} Vega={vega}")
    total_delta += delta

print(f"\n總Delta: {round(total_delta, 4)}")
