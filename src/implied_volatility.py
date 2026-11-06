"""
implied_volatility.py
======================
Inversion du modèle Garman-Kohlhagen pour retrouver la volatilité
implicite à partir d'un prix de marché observé, via la méthode de Brent
(robuste, ne nécessite pas de dérivée analytique comme Newton-Raphson,
et garantit la convergence si un intervalle de bracketing valide existe).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .garman_kohlhagen import FXOption, OptionType


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    option_type: OptionType = "call",
    vol_bounds: tuple[float, float] = (1e-4, 5.0),
) -> float:
    """
    Retourne la volatilité implicite sigma telle que le prix GK(sigma) = market_price.

    Lève une ValueError si le prix de marché est en dehors des bornes
    arbitrables (prix < valeur intrinsèque actualisée ou > borne supérieure),
    ce qui signalerait une opportunité d'arbitrage ou une erreur de saisie.
    """

    def objective(sigma: float) -> float:
        opt = FXOption(S, K, T, sigma, r_d, r_f, option_type)
        return opt.price() - market_price

    lo, hi = vol_bounds
    f_lo, f_hi = objective(lo), objective(hi)

    if f_lo * f_hi > 0:
        raise ValueError(
            f"Impossible de trouver une vol implicite dans [{lo}, {hi}] : "
            f"prix marché={market_price:.6f} hors de l'intervalle de prix "
            f"admissible [{objective(lo) + market_price:.6f}, {objective(hi) + market_price:.6f}]."
        )

    return brentq(objective, lo, hi, xtol=1e-10, maxiter=200)


def implied_vol_surface(
    market_prices: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    S: float,
    r_d: float,
    r_f: float,
    option_type: OptionType = "call",
) -> np.ndarray:
    """
    Construit une surface de vol implicite à partir d'une grille de prix marché.
    market_prices doit avoir la forme (len(maturities), len(strikes)).
    """
    surface = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            surface[i, j] = implied_volatility(
                market_prices[i, j], S, K, T, r_d, r_f, option_type
            )
    return surface
