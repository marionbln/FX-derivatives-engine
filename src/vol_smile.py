"""
vol_smile.py
=============
Modélisation d'un smile/skew de volatilité FX réaliste et évaluation de
son impact sur le pricing par rapport à l'hypothèse (fausse en pratique)
de volatilité plate de Black-Scholes/Garman-Kohlhagen.

On utilise une paramétrisation simple de type "SABR-like" en fonction du
delta / de la moneyness, cohérente avec la convention de marché FX où
les traders cotent en delta (25-delta risk reversal, butterfly, ATM vol)
plutôt qu'en strike directement.

    sigma(K) = sigma_ATM + skew * ln(K/F) + curvature * ln(K/F)^2

C'est une approximation quadratique locale du smile (suffisante pour
illustrer l'effet skew/smile sur le pricing et les Greeks), pas un
modèle de smile arbitrage-free complet (SABR/SVI) — mentionné comme
piste d'extension dans le README.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .garman_kohlhagen import FXOption


def quadratic_smile(K: np.ndarray, F: float, sigma_atm: float, skew: float, curvature: float) -> np.ndarray:
    """
    sigma_atm : vol ATM (at-the-money forward)
    skew : pente du smile (risk reversal-like), typiquement négative sur FX EM,
           reflète l'asymétrie de la distribution des rendements
    curvature : convexité du smile (butterfly-like), toujours >= 0 en pratique
    """
    log_moneyness = np.log(K / F)
    return sigma_atm + skew * log_moneyness + curvature * log_moneyness ** 2


def price_with_smile(
    S, T, r_d, r_f, strikes, sigma_atm, skew, curvature, option_type="call",
) -> pd.DataFrame:
    """
    Compare le pricing avec smile (vol par strike) vs vol plate ATM,
    et montre l'impact sur le prix et le delta pour chaque strike.
    """
    F = S * np.exp((r_d - r_f) * T)  # forward FX
    strikes = np.asarray(strikes)
    smile_vols = quadratic_smile(strikes, F, sigma_atm, skew, curvature)

    rows = []
    for K, vol_k in zip(strikes, smile_vols):
        opt_smile = FXOption(S, K, T, vol_k, r_d, r_f, option_type)
        opt_flat = FXOption(S, K, T, sigma_atm, r_d, r_f, option_type)
        rows.append({
            "strike": K,
            "moneyness_K_over_F": K / F,
            "smile_vol": vol_k,
            "price_smile": opt_smile.price(),
            "price_flat_atm_vol": opt_flat.price(),
            "delta_smile": opt_smile.delta(),
            "delta_flat_atm_vol": opt_flat.delta(),
            "price_diff": opt_smile.price() - opt_flat.price(),
        })
    return pd.DataFrame(rows)
