# Re-import libraries after code state reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

# Black-Scholes model pricing
def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Heston characteristic function
def heston_char_func(phi, S0, K, T, r, kappa, theta, xi, rho, v0, Pnum):
    if Pnum == 1:
        u = 0.5
        b = kappa - rho * xi
    else:
        u = -0.5
        b = kappa

    a = kappa * theta
    x = np.log(S0)
    sigma = xi

    d = np.sqrt((rho * sigma * phi * 1j - b)**2 - sigma**2 * (2 * u * phi * 1j - phi**2))
    g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)

    C = (r * phi * 1j * T + a / sigma**2 * ((b - rho * sigma * phi * 1j + d) * T -
         2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
    D = ((b - rho * sigma * phi * 1j + d) / sigma**2 *
         ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T))))

    return np.exp(C + D * v0 + 1j * phi * x)

# Heston option price calculation
def heston_price(S0, K, T, r, kappa, theta, xi, rho, v0):
    def integrand(phi, Pnum):
        cf = heston_char_func(phi, S0, K, T, r, kappa, theta, xi, rho, v0, Pnum)
        return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))

    P1 = 0.5 + 1 / np.pi * quad(lambda phi: integrand(phi, 1), 0, 100)[0]
    P2 = 0.5 + 1 / np.pi * quad(lambda phi: integrand(phi, 2), 0, 100)[0]

    return S0 * P1 - K * np.exp(-r * T) * P2

# Parameters
S0 = 116.65
T = 200/ 365
r = 0.045
implied_vol = 0.4181
kappa = 1
theta = 0.5672**2
xi = 0.3
rho = -0.7
v0 = 0.4181**2

# Strike range
strikes = np.linspace(100, 130, 50)

# Calculate prices
bs_prices = [bs_price(S0, K, T, r, implied_vol) for K in strikes]
heston_prices = [heston_price(S0, K, T, r, kappa, theta, xi, rho, v0) for K in strikes]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(strikes, bs_prices, label='Black-Scholes', linestyle='--')
plt.plot(strikes, heston_prices, label='Heston', marker='o', markersize=3)
plt.title("Call Price vs Strike: Heston vs Black-Scholes")
plt.xlabel("Strike Price")
plt.ylabel("Call Option Price")
plt.legend()
plt.grid(True)
plt.show()
