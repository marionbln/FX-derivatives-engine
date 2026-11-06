"""
hedging.py
===========
Simulation de couverture en delta dynamique (delta hedging) d'une option
vanille FX vendue, et analyse du P&L de couverture.

Principe : le vendeur de l'option encaisse la prime et couvre son
exposition en delta en achetant/vendant du spot à chaque date de
rebalancement. Le P&L final = prime capitalisée + P&L du portefeuille
de couverture - payoff versé à l'acheteur.

En théorie (couverture continue, pas de coûts de transaction), ce P&L
devrait converger vers 0 en espérance (le prix Black-Scholes/GK est
justement le coût de réplication). En pratique, avec un rebalancement
discret, le P&L résiduel a une variance qui diminue avec la fréquence
de rebalancement (l'effet est illustré ici : c'est l'"error de gamma" /
gamma-theta trade-off).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .garman_kohlhagen import FXOption


@dataclass
class HedgeResult:
    pnl: np.ndarray            # P&L final par trajectoire
    option_price0: float       # prime encaissée à t=0
    rebalance_dates: int
    S_T: np.ndarray


def simulate_delta_hedge(
    S0: float, K: float, T: float, sigma: float, r_d: float, r_f: float,
    option_type: str = "call",
    n_rebalances: int = 21,
    n_paths: int = 10_000,
    seed: int | None = 42,
    realized_sigma: float | None = None,
) -> HedgeResult:
    """
    Simule le P&L d'un vendeur d'option qui delta-hedge dynamiquement.

    realized_sigma : permet de simuler le cas où la vol réalisée du marché
    diffère de la vol implicite utilisée pour pricer/hedger (mismatch vol
    implicite vs réalisée -> source de P&L systématique, concept clé en trading vol).
    Si None, realized_sigma = sigma (cas "cohérent").
    """
    rng = np.random.default_rng(seed)
    realized_sigma = sigma if realized_sigma is None else realized_sigma

    dt = T / n_rebalances
    times = np.linspace(0, T, n_rebalances + 1)

    # simulation des trajectoires sous la vol RÉALISÉE (le monde physique)
    Z = rng.standard_normal((n_paths, n_rebalances))
    drift = (r_d - r_f - 0.5 * realized_sigma ** 2) * dt
    vol_step = realized_sigma * np.sqrt(dt)
    log_incr = drift + vol_step * Z
    log_paths = np.cumsum(log_incr, axis=1)
    log_paths = np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1)
    S = S0 * np.exp(log_paths)

    # prime encaissée à t=0 (pricée avec la vol IMPLICITE, celle du marché)
    opt0 = FXOption(S0, K, T, sigma, r_d, r_f, option_type)
    price0 = opt0.price()

    # cash account en devise domestique : commence avec la prime encaissée
    cash = np.full(n_paths, price0)
    fx_position = np.zeros(n_paths)  # position en devise étrangère (delta)

    for i in range(n_rebalances):
        t_i = times[i]
        tau = T - t_i
        S_i = S[:, i]

        if tau > 1e-8:
            d1 = (np.log(S_i / K) + (r_d - r_f + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
            if option_type == "call":
                delta = np.exp(-r_f * tau) * _norm_cdf(d1)
            else:
                delta = np.exp(-r_f * tau) * (_norm_cdf(d1) - 1)
        else:
            S_T = S[:, -1]
            delta = (S_T > K).astype(float) if option_type == "call" else -(S_T < K).astype(float)

        # coût d'ajustement de la position (achat/vente de spot)
        trade = delta - fx_position
        cash -= trade * S_i  # on paie S_i par unité de devise étrangère achetée
        fx_position = delta

        # capitalisation du cash domestique et de la position étrangère sur dt
        if i < n_rebalances - 1:
            cash *= np.exp(r_d * dt)
            fx_position *= np.exp(r_f * dt)  # la position étrangère génère du r_f

    # à maturité : on liquide la position de couverture au spot final,
    # et on paie le payoff de l'option à l'acheteur
    S_T = S[:, -1]
    cash += fx_position * S_T
    payoff = np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)
    pnl = cash - payoff

    return HedgeResult(pnl=pnl, option_price0=price0, rebalance_dates=n_rebalances, S_T=S_T)


def _norm_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)


def hedging_frequency_study(
    S0, K, T, sigma, r_d, r_f, option_type="call",
    frequencies=(1, 4, 12, 21, 63, 252), n_paths=5000, seed=42,
) -> dict:
    """Étudie comment la variance du P&L de hedging décroît avec la fréquence de rebalancement."""
    results = {}
    for n in frequencies:
        res = simulate_delta_hedge(S0, K, T, sigma, r_d, r_f, option_type, n, n_paths, seed)
        results[n] = {"mean_pnl": res.pnl.mean(), "std_pnl": res.pnl.std(ddof=1)}
    return results
