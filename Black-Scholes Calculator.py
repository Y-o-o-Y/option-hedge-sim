import math
from scipy.stats import norm

def black_scholes_with_greeks(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Prices
    Call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    Put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # Greeks
    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) -
                  r * K * math.exp(-r * T) * norm.cdf(d2))
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) +
                 r * K * math.exp(-r * T) * norm.cdf(-d2))
    rho_call = K * T * math.exp(-r * T) * norm.cdf(d2)
    rho_put = -K * T * math.exp(-r * T) * norm.cdf(-d2)

    # Format output as a clean list
    results = [
        ("Call Price", round(float(Call), 4)),
        ("Put Price", round(float(Put), 4)),
        # ("Call Delta", round(float(delta_call), 4)),
        # ("Put Delta", round(float(delta_put), 4)),
        # ("Gamma", round(float(gamma), 4)), #每1%變動對delta
        # ("Vega (per 1%)", round(float(vega / 100), 4)), # 每1%變動的價格影響
        # ("Call Theta (per day)", round(float(theta_call / 365), 4)),
        # ("Put Theta (per day)", round(float(theta_put / 365), 4)),
        # ("Call Rho (per 1%)", round(float(rho_call / 100), 4)), # 每1%利率變動
        # ("Put Rho (per 1%)", round(float(rho_put / 100), 4))
    ]

    for label, value in results:
        print(f"{label}: {value}")


# print("================================")
# #long
# black_scholes_with_greeks(S=122.8, K=127, T=11/365, r=0.045, sigma=0.3913)
# #short
# black_scholes_with_greeks(S=122.8, K=130, T=11/365, r=0.045, sigma=0.389)

print("================================")
#long
black_scholes_with_greeks(S=129.93, K=130, T=9/365, r=0.045, sigma=0.4181)
#short
# black_scholes_with_greeks(S=122.8, K=113, T=11/365, r=0.045, sigma=0.448)
