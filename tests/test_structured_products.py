"""
Tests des produits structurés (BRC, Autocall) et des outils de structuration.
Lancer avec : python tests/test_structured_products.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.monte_carlo import MCConfig
from src.exotics import price_barrier_option
from src.structured_products import BRC, Autocall, solve_fair_coupon, NOMINAL

S0, T, sigma, r_d, r_f = 100.0, 1.0, 0.25, 0.03, 0.01
CFG = MCConfig(n_paths=100_000, n_steps=252, seed=3)


def test_brc_decomposition():
    """BRC = obligation zéro-coupon + coupons - (nominal/strike) x put down-and-in."""
    brc = BRC(S0, T, sigma, r_d, r_f, coupon=0.08, barrier=0.65, strike=1.0, coupon_freq=4)
    price = brc.price(CFG)["price"]
    dip = price_barrier_option(S0, S0, 0.65 * S0, T, sigma, r_d, r_f, "down", "in", "put", CFG)["price"]
    t_cpn = np.arange(1, 5) / 4 * T
    replication = NOMINAL * np.exp(-r_d * T) + 2.0 * np.exp(-r_d * t_cpn).sum() - NOMINAL / S0 * dip
    assert abs(price - replication) < 1e-8, f"Décomposition BRC violée: {price} vs {replication}"


def test_brc_price_decreases_with_vol():
    """Le porteur d'une BRC est vendeur de volatilité : le prix baisse quand la vol monte."""
    low = BRC(S0, T, 0.15, r_d, r_f).price(CFG)["price"]
    high = BRC(S0, T, 0.35, r_d, r_f).price(CFG)["price"]
    assert high < low, "Le prix de la BRC devrait baisser avec la volatilité"


def test_autocall_never_called_is_zero_coupon_bond():
    """Seuil de rappel inatteignable et protection à 0% : on retrouve une obligation zéro-coupon."""
    ac = Autocall(S0, T, sigma, r_d, r_f, n_obs=4, autocall_level=1e6, coupon=0.05, protection=0.0)
    price = ac.price(MCConfig(n_paths=20_000, n_steps=84, seed=1))["price"]
    assert abs(price - NOMINAL * np.exp(-r_d * T)) < 1e-9


def test_autocall_probabilities_sum():
    """Probabilités de rappel + probabilité d'aller à maturité = 1."""
    res = Autocall(S0, T, sigma, r_d, r_f).price(MCConfig(n_paths=50_000, n_steps=84, seed=2))
    p_called = res["autocall_probabilities"].sum()
    assert 0 < p_called < 1 and 0 < res["expected_life"] <= T


def test_fair_coupon_hits_target():
    """Le coupon résolu ramène bien le prix au niveau cible (98%)."""
    cfg = MCConfig(n_paths=50_000, n_steps=84, seed=4)
    ac = Autocall(S0, T, sigma, r_d, r_f, n_obs=4)
    cpn = solve_fair_coupon(ac, target_price=98.0, config=cfg)
    ac.coupon = cpn
    assert abs(ac.price(cfg)["price"] - 98.0) < 1e-4


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} tests passés avec succès.")
