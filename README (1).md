# Options Portfolio Construction & Risk Analytics: HDFC Bank

Python project from the Financial Risk Analytics and Management course (BITS Pilani): option pricing, Greeks, hedging and Value at Risk on HDFC Bank (NSE: HDFCBANK).

## What it does
1. **Return analysis:** daily log returns from Yahoo Finance prices (2020 onwards); annualised volatility, skewness and excess kurtosis.
2. **Black-Scholes pricing:** prices for 3-month at-the-money calls and puts, with all Greeks (delta, gamma, vega, theta, rho).
3. **Portfolio:** long 1,000 ATM calls, short 500 puts at 95% of spot, short 500 calls at 105% of spot.
4. **Hedging:** gamma hedge with a 1-month ATM call, then delta hedge with HDFC Bank shares, giving a delta- and gamma-neutral book.
5. **Value at Risk:** 1-day 95% and 99% VaR by historical simulation (last 500 days) with full Black-Scholes revaluation, for the unhedged and hedged book, plus parametric VaR for comparison.

## How to run
```
pip install -r requirements.txt
python options_risk.py
```

## Assumptions and limitations
- Risk-free rate fixed at 6.5%; volatility is historical and constant.
- Black-Scholes assumes normal returns, while real returns show skew and fat tails.
- Hedges are static (set once, not rebalanced daily).
