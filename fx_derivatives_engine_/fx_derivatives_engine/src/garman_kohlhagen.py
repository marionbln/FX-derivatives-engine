"""
garman_kohlhagen.py
====================
Pricing analytique d'options vanilles FX (modèle Garman-Kohlhagen,
l'extension de Black-Scholes aux devises avec deux taux d'intérêt :
taux domestique r_d et taux étranger r_f, qui joue le rôle du
"dividend yield" continu).

Convention : S = spot (nombre d'unités de devise domestique pour 1 unité
de devise étrangère), K = strike, T = maturité en années,
sigma = volatilité annualisée, r_d = taux domestique, r_f = taux étranger.

Formules :
    d1 = [ln(S/K) + (r_d - r_f + 0.5*sigma^2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S * exp(-r_f*T) * N(d1) - K * exp(-r_d*T) * N(d2)
    Put  = K * exp(-r_d*T) * N(-d2) - S * exp(-r_f*T) * N(-d1)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass
class FXOption:
    """Spécification d'une option vanille FX européenne."""
    S: float      # spot
    K: float      # strike
    T: float      # maturité (années)
    sigma: float  # volatilité annualisée
    r_d: float    # taux domestique (continu)
    r_f: float    # taux étranger (continu)
    option_type: OptionType = "call"

    def _d1_d2(self) -> tuple[float, float]:
        if self.T <= 0 or self.sigma <= 0:
            raise ValueError("T et sigma doivent être strictement positifs")
        d1 = (np.log(self.S / self.K) + (self.r_d - self.r_f + 0.5 * self.sigma ** 2) * self.T) \
             / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    # ------------------------------------------------------------------
    # Prix
    # ------------------------------------------------------------------
    def price(self) -> float:
        d1, d2 = self._d1_d2()
        S, K, T, r_d, r_f = self.S, self.K, self.T, self.r_d, self.r_f
        if self.option_type == "call":
            return S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
        elif self.option_type == "put":
            return K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)
        raise ValueError("option_type doit être 'call' ou 'put'")

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------
    def delta(self) -> float:
        d1, _ = self._d1_d2()
        disc_f = np.exp(-self.r_f * self.T)
        if self.option_type == "call":
            return disc_f * norm.cdf(d1)
        return disc_f * (norm.cdf(d1) - 1)

    def gamma(self) -> float:
        d1, _ = self._d1_d2()
        disc_f = np.exp(-self.r_f * self.T)
        return disc_f * norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        """Sensibilité au sigma, pour 1.00 (100%) de vol. Diviser par 100 pour 1 vol point."""
        d1, _ = self._d1_d2()
        disc_f = np.exp(-self.r_f * self.T)
        return self.S * disc_f * norm.pdf(d1) * np.sqrt(self.T)

    def theta(self) -> float:
        """Sensibilité au temps qui passe (par an). Diviser par 365 pour theta/jour."""
        d1, d2 = self._d1_d2()
        S, K, T, sigma, r_d, r_f = self.S, self.K, self.T, self.sigma, self.r_d, self.r_f
        disc_d, disc_f = np.exp(-r_d * T), np.exp(-r_f * T)
        term_common = -S * disc_f * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if self.option_type == "call":
            return term_common + r_f * S * disc_f * norm.cdf(d1) - r_d * K * disc_d * norm.cdf(d2)
        return term_common - r_f * S * disc_f * norm.cdf(-d1) + r_d * K * disc_d * norm.cdf(-d2)

    def rho_domestic(self) -> float:
        d1, d2 = self._d1_d2()
        disc_d = np.exp(-self.r_d * self.T)
        if self.option_type == "call":
            return self.K * self.T * disc_d * norm.cdf(d2)
        return -self.K * self.T * disc_d * norm.cdf(-d2)

    def rho_foreign(self) -> float:
        d1, _ = self._d1_d2()
        disc_f = np.exp(-self.r_f * self.T)
        if self.option_type == "call":
            return -self.S * self.T * disc_f * norm.cdf(d1)
        return self.S * self.T * disc_f * norm.cdf(-d1)

    def greeks(self) -> dict:
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega() / 100,      # pour 1 point de vol (1%)
            "theta": self.theta() / 365,    # par jour
            "rho_domestic": self.rho_domestic() / 100,
            "rho_foreign": self.rho_foreign() / 100,
        }


def gk_price(S, K, T, sigma, r_d, r_f, option_type: OptionType = "call") -> float:
    """Fonction utilitaire rapide (sans dataclass) pour un prix GK."""
    return FXOption(S, K, T, sigma, r_d, r_f, option_type).price()
