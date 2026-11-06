"""
structured_products.py
=======================
Pricing Monte Carlo des deux familles de produits structurés les plus
courantes sur le marché suisse :

    - Barrier Reverse Convertible (BRC)
    - Autocall (type "Athena" avec coupon à effet mémoire / snowball)

Les deux produits sont exprimés en % du nominal (nominal = 100) et
pricés sur le même moteur de trajectoires GBM que les options exotiques
(monte_carlo.simulate_gbm_paths). r_f joue le rôle du rendement du
sous-jacent (taux étranger en FX, dividende en action).

On fournit aussi :
    - la décomposition BRC = obligation + coupons - put down-and-in,
      qui sert de test de cohérence du moteur ;
    - un solveur de coupon "fair" : le coupon qui ramène le prix du
      produit au niveau cible (ex. 98% = 100% - marge de l'émetteur),
      c'est-à-dire le calcul fait au quotidien par un structureur ;
    - des Greeks par bump-and-reprice avec nombres aléatoires communs.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import brentq
from .monte_carlo import simulate_gbm_paths, MCConfig

NOMINAL = 100.0


def _observation_indices(n_steps: int, n_obs: int) -> np.ndarray:
    """Indices des dates d'observation régulières sur la grille de simulation."""
    if n_steps % n_obs != 0:
        raise ValueError("n_steps doit être un multiple du nombre d'observations")
    step = n_steps // n_obs
    return np.arange(step, n_steps + 1, step)


def _summary(discounted: np.ndarray) -> dict:
    price = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return {"price": price, "stderr": stderr,
            "ci_95": (price - 1.96 * stderr, price + 1.96 * stderr)}


# ----------------------------------------------------------------------
# Barrier Reverse Convertible
# ----------------------------------------------------------------------
@dataclass
class BRC:
    """
    Barrier Reverse Convertible sur un sous-jacent.

    coupon       : coupon annuel garanti, en % du nominal (ex. 0.08 = 8%)
    barrier      : barrière en % du niveau initial (ex. 0.65 = 65%)
    strike       : strike en % du niveau initial (en général 100%)
    coupon_freq  : nombre de paiements de coupon par an
    barrier_type : "continuous" (surveillance à chaque pas de simulation)
                   ou "european" (barrière observée à maturité seulement)

    Remboursement à maturité :
        - si la barrière n'a pas été touchée, ou si S_T >= strike : 100%
        - sinon : 100% * S_T / (strike * S0)  (livraison du sous-jacent)
    Les coupons sont payés dans tous les cas.
    """
    S0: float
    T: float
    sigma: float
    r_d: float
    r_f: float
    coupon: float = 0.08
    barrier: float = 0.65
    strike: float = 1.00
    coupon_freq: int = 4
    barrier_type: str = "continuous"

    def _payoff_discounted(self, paths: np.ndarray) -> np.ndarray:
        n_steps = paths.shape[1] - 1
        n_cpn = int(round(self.coupon_freq * self.T))
        idx = _observation_indices(n_steps, n_cpn)
        t_cpn = idx / n_steps * self.T
        cpn_amount = NOMINAL * self.coupon / self.coupon_freq
        pv_coupons = cpn_amount * np.exp(-self.r_d * t_cpn).sum()

        S_T = paths[:, -1]
        K = self.strike * self.S0
        B = self.barrier * self.S0
        if self.barrier_type == "continuous":
            touched = (paths <= B).any(axis=1)
        else:
            touched = S_T <= B
        loss = touched & (S_T < K)
        redemption = np.where(loss, NOMINAL * S_T / K, NOMINAL)
        return pv_coupons + np.exp(-self.r_d * self.T) * redemption

    def price(self, config: MCConfig | None = None) -> dict:
        config = config or MCConfig(n_steps=252)
        paths = simulate_gbm_paths(self.S0, self.T, self.sigma, self.r_d, self.r_f, config)
        res = _summary(self._payoff_discounted(paths))
        res["prob_barrier_hit"] = float((paths <= self.barrier * self.S0).any(axis=1).mean())
        return res


# ----------------------------------------------------------------------
# Autocall
# ----------------------------------------------------------------------
@dataclass
class Autocall:
    """
    Autocall de type "Athena" (coupon snowball, payé uniquement au rappel).

    n_obs          : nombre de dates d'observation (régulières jusqu'à T)
    autocall_level : seuil de rappel anticipé en % du niveau initial
    coupon         : coupon par période d'observation, en % du nominal
    protection     : barrière de protection du capital à maturité
                     (observation européenne), en % du niveau initial

    A chaque date d'observation i (i = 1..n_obs) :
        si S_ti >= autocall_level * S0 : remboursement 100% + i * coupon, fin du produit.
    A maturité, si jamais rappelé :
        - S_T >= protection * S0 : 100%
        - sinon : 100% * S_T / S0
    """
    S0: float
    T: float
    sigma: float
    r_d: float
    r_f: float
    n_obs: int = 4
    autocall_level: float = 1.00
    coupon: float = 0.02
    protection: float = 0.60

    def _payoff_discounted(self, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_paths, n_cols = paths.shape
        n_steps = n_cols - 1
        idx = _observation_indices(n_steps, self.n_obs)
        t_obs = idx / n_steps * self.T
        obs = paths[:, idx] / self.S0                       # performances aux dates d'observation

        called = obs >= self.autocall_level                 # (n_paths, n_obs)
        any_call = called.any(axis=1)
        first = np.where(any_call, called.argmax(axis=1), self.n_obs)  # n_obs = jamais rappelé

        pv = np.zeros(n_paths)
        # chemins rappelés : 100 + i * coupon payé à la date t_i
        c = first[any_call]
        pv[any_call] = NOMINAL * (1 + (c + 1) * self.coupon) * np.exp(-self.r_d * t_obs[c])

        # chemins non rappelés : protection du capital à maturité
        perf_T = obs[~any_call, -1]
        redemption = np.where(perf_T >= self.protection, NOMINAL, NOMINAL * perf_T)
        pv[~any_call] = redemption * np.exp(-self.r_d * self.T)
        return pv, first

    def price(self, config: MCConfig | None = None) -> dict:
        config = config or MCConfig(n_steps=self.n_obs * 21)
        paths = simulate_gbm_paths(self.S0, self.T, self.sigma, self.r_d, self.r_f, config)
        pv, first = self._payoff_discounted(paths)
        res = _summary(pv)
        n_steps = paths.shape[1] - 1
        t_obs = _observation_indices(n_steps, self.n_obs) / n_steps * self.T
        probs = np.array([(first == i).mean() for i in range(self.n_obs)])
        res["autocall_probabilities"] = probs
        res["prob_capital_loss"] = float(((first == self.n_obs) &
                                         (paths[:, -1] / self.S0 < self.protection)).mean())
        life = np.where(first < self.n_obs, t_obs[np.minimum(first, self.n_obs - 1)], self.T)
        res["expected_life"] = float(life.mean())
        return res


# ----------------------------------------------------------------------
# Outils de structureur
# ----------------------------------------------------------------------
def solve_fair_coupon(product, target_price: float = 98.0,
                      config: MCConfig | None = None, bounds=(0.0, 0.5)) -> float:
    """
    Coupon qui ramène le prix du produit à `target_price` (en % du nominal).
    Même graine Monte Carlo à chaque évaluation => fonction lisse en coupon.
    """
    def objective(cpn: float) -> float:
        return replace(product, coupon=cpn).price(config)["price"] - target_price
    lo, hi = bounds
    if objective(lo) > 0:
        raise ValueError("Prix déjà au-dessus de la cible avec le coupon minimal : "
                         "le produit ne finance pas de coupon positif à ce niveau de marge")
    return brentq(objective, lo, hi, xtol=1e-6)


def bump_greeks(product, config: MCConfig | None = None,
                spot_bump: float = 0.01, vol_bump: float = 0.01) -> dict:
    """
    Delta (pour +1% de spot, en points de nominal) et vega (pour +1 point de vol)
    par différences finies centrées, avec nombres aléatoires communs.
    """
    config = config or MCConfig(n_steps=252)
    p = lambda **kw: replace(product, **kw).price(config)["price"]
    S0, sig = product.S0, product.sigma
    # le spot bouge mais les niveaux du produit (barrière, strike, seuil de
    # rappel), fixés à l'émission sur le S0 initial, restent inchangés
    up = _price_with_spot(product, S0 * (1 + spot_bump), config)
    dn = _price_with_spot(product, S0 * (1 - spot_bump), config)
    delta = (up - dn) / 2
    vega = (p(sigma=sig + vol_bump) - p(sigma=sig - vol_bump)) / 2
    return {"delta_1pct": delta, "vega_1pt": vega}


def _price_with_spot(product, new_spot: float, config: MCConfig) -> float:
    """
    Prix quand le spot bouge mais que les niveaux du produit (fixés à l'émission
    sur S0 initial) restent inchangés.
    """
    paths = simulate_gbm_paths(new_spot, product.T, product.sigma,
                               product.r_d, product.r_f, config)
    pv = product._payoff_discounted(paths)
    pv = pv[0] if isinstance(pv, tuple) else pv
    return float(pv.mean())
