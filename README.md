# FX Derivatives Pricing & Hedging Engine

A Python engine for pricing and risk-managing options and structured products:
from a closed-form FX vanilla pricer to Monte Carlo exotics, dynamic delta
hedging, the volatility smile, and the two most common Swiss structured
products: **Barrier Reverse Convertibles** and **Autocalls**.

*Version française : [README.fr.md](README.fr.md)*

![Delta hedging P&L](outputs/04_delta_hedging.png)

## Features

| Module | What it does |
|---|---|
| **Pricing & Greeks** (`garman_kohlhagen.py`) | Garman-Kohlhagen model (Black-Scholes for FX, with domestic and foreign rates). Price, delta, gamma, vega, theta, domestic and foreign rho. |
| **Implied volatility** (`implied_volatility.py`) | Inverts the model to recover the volatility from an observed price (Brent's method, with no-arbitrage bounds checks). |
| **Monte Carlo** (`monte_carlo.py`) | Exact GBM simulation under the domestic risk-neutral measure, antithetic variates, 95% confidence intervals. Validated against the closed-form price. |
| **Exotic options** (`exotics.py`) | Barrier options (up/down, knock-in/knock-out, discrete monitoring) and Asian options (arithmetic and geometric averages), priced by Monte Carlo. |
| **Delta hedging** (`hedging.py`) | Simulates a dealer who sells an option and delta-hedges it. Shows how hedging P&L variance falls with rebalancing frequency, and the P&L created when realised volatility differs from implied volatility. |
| **Sensitivities** (`sensitivity.py`) | Price as a function of spot, volatility and time to maturity. |
| **Volatility smile** (`vol_smile.py`) | Quadratic smile in log-moneyness and the pricing/delta error made when a flat ATM volatility is used for every strike. |
| **Structured products** (`structured_products.py`) | Barrier Reverse Convertible (continuous or European barrier, physical-delivery equivalent) and Athena-style Autocall (snowball coupon, capital protection barrier). Autocall probabilities, expected life, probability of capital loss. |
| **Structuring tools** (`structured_products.py`) | Fair-coupon solver (coupon that brings the product to a target price, e.g. 98% = 100% minus issuer margin) and bump-and-reprice delta and vega with common random numbers. |

## Results

![Structured products](outputs/07_structured_products.png)

*Left: fair BRC coupon vs barrier level. Middle: autocall probability at each observation date. Right: fair autocall coupon vs volatility (target price 98%).*

| Monte Carlo vs closed form | Volatility smile |
|---|---|
| ![MC convergence](outputs/02_mc_convergence.png) | ![Vol smile](outputs/06_vol_smile.png) |

| Pricing & Greeks | Exotic options |
|---|---|
| ![Greeks](outputs/01_pricing_greeks.png) | ![Exotics](outputs/03_exotics.png) |

## Validation

`tests/test_pricing.py` checks the engine against known relationships:

- put-call parity on the closed-form prices;
- implied volatility round-trip (price -> vol -> price);
- Monte Carlo price converging to the Garman-Kohlhagen price;
- barrier in-out parity (knock-in + knock-out = vanilla);
- delta bounds and positive gamma.

`tests/test_structured_products.py` checks the structured products:

- BRC replication: zero-coupon bond + coupons - down-and-in put (exact on the same paths);
- BRC price decreases with volatility (the investor is short volatility);
- an autocall that can never be called with no protection barrier prices as a zero-coupon bond;
- autocall probabilities and expected life are consistent;
- the fair-coupon solver hits the target price.

All 11 tests pass.

## Project structure

- `src/` : pricing engine (`garman_kohlhagen.py`, `implied_volatility.py`, `monte_carlo.py`, `exotics.py`, `hedging.py`, `sensitivity.py`, `vol_smile.py`, `structured_products.py`)
- `examples/` : one script per module, generates the charts
- `outputs/` : generated charts
- `tests/` : validation tests

## Getting started

    pip install -r requirements.txt
    python tests/test_pricing.py
    python tests/test_structured_products.py
    python examples/01_pricing_and_greeks.py

Each script in `examples/` reproduces one chart in `outputs/`.

## Limitations and next steps

- Exotics are priced by simulation only (no closed-form barrier formulas yet).
- The smile is a simple quadratic parametrisation, not an arbitrage-free model such as SABR or SVI.
- The hedging simulation ignores transaction costs.
- Structured products are priced on a single underlying with flat volatility; worst-of baskets (correlated paths) and a local/stochastic volatility model would be the natural next steps.

## Tools

Python, NumPy, SciPy, Pandas, Matplotlib.
