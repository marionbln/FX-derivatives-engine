"""
sensitivity.py
================
Analyse de sensibilité du prix et des Greeks d'une option FX par rapport
au spot, à la vol et à la maturité — utile pour du "what-if" / stress
testing façon desk de trading ou market risk (scénarios de choc de marché).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .garman_kohlhagen import FXOption


def sensitivity_grid(
    K, T, sigma, r_d, r_f, option_type="call",
    spot_range=None, vol_range=None,
) -> pd.DataFrame:
    """Grille prix/Greeks en fonction du spot pour une vol/maturité donnée."""
    if spot_range is None:
        spot_range = np.linspace(K * 0.7, K * 1.3, 61)
    rows = []
    for S in spot_range:
        opt = FXOption(S, K, T, sigma, r_d, r_f, option_type)
        g = opt.greeks()
        g["spot"] = S
        rows.append(g)
    return pd.DataFrame(rows).set_index("spot")


def price_surface_spot_vol(
    K, T, r_d, r_f, option_type="call",
    spot_range=None, vol_range=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surface de prix en fonction de (spot, vol) pour une maturité fixée."""
    if spot_range is None:
        spot_range = np.linspace(K * 0.7, K * 1.3, 40)
    if vol_range is None:
        vol_range = np.linspace(0.02, 0.30, 40)

    Z = np.zeros((len(vol_range), len(spot_range)))
    for i, sigma in enumerate(vol_range):
        for j, S in enumerate(spot_range):
            Z[i, j] = FXOption(S, K, T, sigma, r_d, r_f, option_type).price()
    return spot_range, vol_range, Z


def maturity_decay(
    S, K, sigma, r_d, r_f, option_type="call", maturities=None,
) -> pd.DataFrame:
    """Étudie la décroissance du prix et du theta en fonction du temps restant (time decay)."""
    if maturities is None:
        maturities = np.linspace(1 / 365, 2.0, 100)
    rows = []
    for T in maturities:
        opt = FXOption(S, K, T, sigma, r_d, r_f, option_type)
        g = opt.greeks()
        g["T"] = T
        rows.append(g)
    return pd.DataFrame(rows).set_index("T")
