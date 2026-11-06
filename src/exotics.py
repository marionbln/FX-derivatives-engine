"""
exotics.py
===========
Pricing d'options exotiques FX (path-dependent) par Monte Carlo :
    - Options à barrière (Knock-Out / Knock-In, Up/Down)
    - Options asiatiques (moyenne arithmétique, strike fixe)

Ces produits n'ont généralement pas de formule fermée simple (ou une
formule fermée complexe/approximative pour l'asiatique arithmétique),
donc la simulation Monte Carlo sur trajectoires complètes est l'approche
standard utilisée en desk pour les pricer et pour calculer leurs Greeks
(par différences finies / bump-and-reprice).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from .monte_carlo import simulate_gbm_paths, MCConfig

BarrierDirection = Literal["up", "down"]
BarrierKind = Literal["in", "out"]
OptionType = Literal["call", "put"]


def price_barrier_option(
    S0: float, K: float, barrier: float, T: float, sigma: float, r_d: float, r_f: float,
    direction: BarrierDirection, kind: BarrierKind, option_type: OptionType = "call",
    config: MCConfig | None = None,
) -> dict:
    """
    Prix Monte Carlo d'une option à barrière européenne.

    direction='up'   -> barrière touchée si le spot dépasse `barrier`
    direction='down' -> barrière touchée si le spot passe sous `barrier`
    kind='out'        -> l'option est annulée (knock-out) si la barrière est touchée
    kind='in'          -> l'option n'existe (knock-in) que si la barrière est touchée

    Note : la barrière est surveillée sur les points de discrétisation simulés
    (surveillance discrète), ce qui est l'approche standard en pratique de desk
    (fixing quotidien) ; plus n_steps est grand, plus on se rapproche du continu.
    """
    config = config or MCConfig()
    paths = simulate_gbm_paths(S0, T, sigma, r_d, r_f, config)
    S_T = paths[:, -1]

    if direction == "up":
        touched = (paths >= barrier).any(axis=1)
    else:
        touched = (paths <= barrier).any(axis=1)

    if option_type == "call":
        vanilla_payoff = np.maximum(S_T - K, 0.0)
    else:
        vanilla_payoff = np.maximum(K - S_T, 0.0)

    if kind == "out":
        payoff = np.where(touched, 0.0, vanilla_payoff)
    else:  # 'in'
        payoff = np.where(touched, vanilla_payoff, 0.0)

    discounted = np.exp(-r_d * T) * payoff
    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return {
        "price": price,
        "stderr": stderr,
        "ci_95": (price - 1.96 * stderr, price + 1.96 * stderr),
        "prob_touch": touched.mean(),
    }


def price_asian_option(
    S0: float, K: float, T: float, sigma: float, r_d: float, r_f: float,
    option_type: OptionType = "call", config: MCConfig | None = None,
    averaging: Literal["arithmetic", "geometric"] = "arithmetic",
) -> dict:
    """
    Prix Monte Carlo d'une option asiatique à strike fixe, moyenne calculée
    sur tous les points de discrétisation simulés (fixings réguliers).
    """
    config = config or MCConfig()
    paths = simulate_gbm_paths(S0, T, sigma, r_d, r_f, config)
    # on exclut S0 (t=0) de la moyenne, convention "fixings sur la vie de l'option"
    observations = paths[:, 1:]

    if averaging == "arithmetic":
        avg = observations.mean(axis=1)
    else:
        avg = np.exp(np.log(observations).mean(axis=1))

    if option_type == "call":
        payoff = np.maximum(avg - K, 0.0)
    else:
        payoff = np.maximum(K - avg, 0.0)

    discounted = np.exp(-r_d * T) * payoff
    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return {
        "price": price,
        "stderr": stderr,
        "ci_95": (price - 1.96 * stderr, price + 1.96 * stderr),
    }


def barrier_parity_check(
    S0: float, K: float, barrier: float, T: float, sigma: float, r_d: float, r_f: float,
    direction: BarrierDirection, option_type: OptionType = "call", config: MCConfig | None = None,
) -> dict:
    """
    Vérifie la relation de parité Knock-In + Knock-Out = Vanille (même strike/barrière),
    utile comme test de cohérence du moteur de simulation.
    """
    config = config or MCConfig()
    ko = price_barrier_option(S0, K, barrier, T, sigma, r_d, r_f, direction, "out", option_type, config)
    ki = price_barrier_option(S0, K, barrier, T, sigma, r_d, r_f, direction, "in", option_type, config)
    from .monte_carlo import mc_price_vanilla
    vanilla = mc_price_vanilla(S0, K, T, sigma, r_d, r_f, option_type, config)
    return {
        "knock_out": ko["price"],
        "knock_in": ki["price"],
        "sum_in_out": ko["price"] + ki["price"],
        "vanilla": vanilla["price"],
        "diff": abs(ko["price"] + ki["price"] - vanilla["price"]),
    }
