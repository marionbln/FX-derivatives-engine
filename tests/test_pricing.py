"""
Tests unitaires de base : parité put-call, convergence MC, parité barrières,
recouvrement de la vol implicite. Lancer avec : python -m pytest tests/ -v
(ou simplement : python tests/test_pricing.py)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.garman_kohlhagen import FXOption
from src.implied_volatility import implied_volatility
from src.monte_carlo import mc_price_vanilla, MCConfig
from src.exotics import barrier_parity_check

S0, K, T, sigma, r_d, r_f = 1.10, 1.08, 0.75, 0.11, 0.04, 0.025


def test_put_call_parity():
    c = FXOption(S0, K, T, sigma, r_d, r_f, "call").price()
    p = FXOption(S0, K, T, sigma, r_d, r_f, "put").price()
    lhs = c - p
    rhs = S0 * np.exp(-r_f * T) - K * np.exp(-r_d * T)
    assert abs(lhs - rhs) < 1e-10, f"Parité put-call violée: {lhs} vs {rhs}"


def test_implied_vol_roundtrip():
    price = FXOption(S0, K, T, sigma, r_d, r_f, "call").price()
    iv = implied_volatility(price, S0, K, T, r_d, r_f, "call")
    assert abs(iv - sigma) < 1e-6, f"Vol implicite non retrouvée: {iv} vs {sigma}"


def test_monte_carlo_matches_analytical():
    analytical = FXOption(S0, K, T, sigma, r_d, r_f, "call").price()
    mc = mc_price_vanilla(S0, K, T, sigma, r_d, r_f, "call", MCConfig(n_paths=300_000, n_steps=1, seed=0))
    lo, hi = mc["ci_95"]
    assert lo <= analytical <= hi, f"MC hors IC95%: analytique={analytical}, IC=({lo},{hi})"


def test_barrier_in_out_parity():
    res = barrier_parity_check(S0, K, 1.20, T, sigma, r_d, r_f, "up", "call",
                                 MCConfig(n_paths=100_000, n_steps=60, seed=1))
    assert res["diff"] < 1e-9, f"Parité KI+KO != vanille: diff={res['diff']}"


def test_delta_bounds():
    call_delta = FXOption(S0, K, T, sigma, r_d, r_f, "call").delta()
    put_delta = FXOption(S0, K, T, sigma, r_d, r_f, "put").delta()
    assert 0 <= call_delta <= 1, "Delta call hors bornes [0,1]"
    assert -1 <= put_delta <= 0, "Delta put hors bornes [-1,0]"


def test_gamma_positive():
    for ot in ("call", "put"):
        g = FXOption(S0, K, T, sigma, r_d, r_f, ot).gamma()
        assert g > 0, f"Gamma doit être positif ({ot})"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} tests passés avec succès.")
