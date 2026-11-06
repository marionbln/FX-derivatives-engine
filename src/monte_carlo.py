"""
monte_carlo.py
================
Moteur Monte Carlo pour la simulation de trajectoires de spot FX sous la
mesure risque-neutre domestique, et pricing par simulation (vanilles +
sert de brique de base pour les options exotiques dans exotics.py).

Dynamique simulée (discrétisation exacte du GBM, pas de biais de schéma) :
    S_{t+dt} = S_t * exp[ (r_d - r_f - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z ],  Z ~ N(0,1)

Réduction de variance : variables antithétiques (Z et -Z), qui réduisent
la variance de l'estimateur sans biais et sans coût de calcul significatif.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MCConfig:
    n_paths: int = 100_000
    n_steps: int = 100
    antithetic: bool = True
    seed: int | None = 42


def simulate_gbm_paths(
    S0: float, T: float, sigma: float, r_d: float, r_f: float, config: MCConfig
) -> np.ndarray:
    """
    Simule des trajectoires de spot GBM sous mesure risque-neutre domestique.
    Retourne un array de forme (n_paths_total, n_steps+1), colonne 0 = S0.
    """
    rng = np.random.default_rng(config.seed)
    dt = T / config.n_steps
    drift = (r_d - r_f - 0.5 * sigma ** 2) * dt
    vol_step = sigma * np.sqrt(dt)

    n_half = config.n_paths // 2 if config.antithetic else config.n_paths
    Z = rng.standard_normal((n_half, config.n_steps))
    if config.antithetic:
        Z = np.concatenate([Z, -Z], axis=0)

    log_increments = drift + vol_step * Z
    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.concatenate([np.zeros((log_paths.shape[0], 1)), log_paths], axis=1)
    paths = S0 * np.exp(log_paths)
    return paths


def mc_price_vanilla(
    S0: float, K: float, T: float, sigma: float, r_d: float, r_f: float,
    option_type: str = "call", config: MCConfig | None = None,
) -> dict:
    """Prix Monte Carlo d'une option vanille + intervalle de confiance à 95%."""
    config = config or MCConfig()
    paths = simulate_gbm_paths(S0, T, sigma, r_d, r_f, config)
    S_T = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r_d * T) * payoffs
    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return {
        "price": price,
        "stderr": stderr,
        "ci_95": (price - 1.96 * stderr, price + 1.96 * stderr),
        "n_paths": len(discounted),
    }
