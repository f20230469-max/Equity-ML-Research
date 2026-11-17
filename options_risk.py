# Options Portfolio Construction & Risk Analytics - HDFC Bank (NSE)
# pip install yfinance pandas numpy scipy

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm, skew, kurtosis

TICKER = "HDFCBANK.NS"
START = "2020-01-01"
R = 0.065          # risk-free rate (approx. Indian T-bill yield)
T_MAIN = 90 / 365  # 3-month options
T_HEDGE = 30 / 365 # 1-month option used for the gamma hedge

# ---------------- 1. Data and return statistics ----------------
raw = yf.download(TICKER, start=START, auto_adjust=True, progress=False)
close = raw["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]
close = close.dropna()
log_ret = np.log(close / close.shift(1)).dropna()

sigma = log_ret.std() * np.sqrt(252)
S = float(close.iloc[-1])
print(f"Spot price: {S:.2f}")
print(f"Annualized volatility: {sigma:.2%}")
print(f"Annualized mean return: {log_ret.mean() * 252:.2%}")
print(f"Skewness: {skew(log_ret):.3f} | Excess kurtosis: {kurtosis(log_ret):.3f}")


# ---------------- 2. Black-Scholes price and Greeks ----------------
def bs(S, K, T, r, sigma, kind="call"):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


K_atm = round(S, -1)
print("\nATM 3-month option prices and Greeks (per share):")
greeks = pd.DataFrame({k: bs(S, K_atm, T_MAIN, R, sigma, k) for k in ["call", "put"]}).T
print(greeks.round(4))

# ---------------- 3. Portfolio and delta-gamma hedge ----------------
# Positions: (kind, strike, expiry, quantity in shares; negative = short)
portfolio = [("call", K_atm, T_MAIN, 1000),
             ("put", round(S * 0.95, -1), T_MAIN, -500),
             ("call", round(S * 1.05, -1), T_MAIN, -500)]


def book_greeks(S, positions, T_shift=0.0):
    tot = {"price": 0.0, "delta": 0.0, "gamma": 0.0}
    for kind, K, T, q in positions:
        g = bs(S, K, max(T - T_shift, 1e-6), R, sigma, kind)
        for k in tot:
            tot[k] += q * g[k]
    return tot


before = book_greeks(S, portfolio)
print(f"\nUnhedged book: value {before['price']:,.0f} | delta {before['delta']:,.1f} | gamma {before['gamma']:,.3f}")

# Gamma hedge with a 1-month ATM call, then delta hedge with the stock
h = bs(S, K_atm, T_HEDGE, R, sigma, "call")
q_hedge = -before["gamma"] / h["gamma"]
hedged = portfolio + [("call", K_atm, T_HEDGE, q_hedge)]
after_opt = book_greeks(S, hedged)
q_stock = -after_opt["delta"]
print(f"Gamma hedge: {q_hedge:,.1f} units of 1-month ATM call | Delta hedge: {q_stock:,.1f} shares")


def book_value(S_new, positions, stock_qty, T_shift):
    return book_greeks(S_new, positions, T_shift)["price"] + stock_qty * S_new


final = book_greeks(S, hedged)
print(f"Hedged book: delta {final['delta'] + q_stock:,.3f} | gamma {final['gamma']:,.5f}")

# ---------------- 4. Value at Risk (1-day, historical full revaluation) ----------------
scenarios = S * np.exp(log_ret.values[-500:])   # last 500 daily moves applied to today's price
dt = 1 / 365
for name, pos, stock in [("Unhedged", portfolio, 0.0), ("Delta-gamma hedged", hedged, q_stock)]:
    v0 = book_value(S, pos, stock, 0.0)
    pnl = np.array([book_value(s, pos, stock, dt) - v0 for s in scenarios])
    var95, var99 = -np.percentile(pnl, 5), -np.percentile(pnl, 1)
    print(f"{name:<20} 1-day VaR 95%: {var95:,.0f} | 99%: {var99:,.0f}")

# Parametric (delta-normal) VaR of the stock alone, for comparison
daily_sd = log_ret.std()
print(f"\nParametric 1-day VaR of 1 share: 95% {1.645 * daily_sd * S:.2f} | 99% {2.326 * daily_sd * S:.2f}")
