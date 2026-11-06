"""
Exemple 7 : Produits structurés (BRC et Autocall) vus par un structureur.
    - coupon "fair" d'une BRC en fonction du niveau de barrière
    - probabilités de rappel d'un autocall à chaque date d'observation
    - coupon "fair" d'un autocall en fonction de la volatilité
Prix cible : 98% du nominal (100% - 2% de marge émetteur).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from src.monte_carlo import MCConfig
from src.structured_products import BRC, Autocall, solve_fair_coupon, bump_greeks

S0, T, r_d, r_f = 100.0, 1.0, 0.03, 0.01
sigma = 0.25
TARGET = 98.0
cfg = MCConfig(n_paths=60_000, n_steps=252, seed=11)

# --- BRC : coupon fair vs barrière ---
barriers = np.linspace(0.50, 0.90, 9)
brc_coupons = [solve_fair_coupon(BRC(S0, T, sigma, r_d, r_f, barrier=b), TARGET, cfg) for b in barriers]

# --- Autocall : produit de référence (2 ans, observations trimestrielles) ---
cfg_ac = MCConfig(n_paths=60_000, n_steps=168, seed=11)
ac = Autocall(S0, 2.0, sigma, r_d, r_f, n_obs=8, autocall_level=1.0, protection=0.60)
ac.coupon = solve_fair_coupon(ac, TARGET, cfg_ac)
res = ac.price(cfg_ac)
greeks = bump_greeks(ac, cfg_ac)
print(f"Autocall 2 ans, 8 observations, protection 60%, vol {sigma:.0%}")
print(f"  coupon fair par trimestre : {ac.coupon:.2%}  (prix {res['price']:.2f}%)")
print(f"  durée de vie moyenne      : {res['expected_life']:.2f} an")
print(f"  proba de perte en capital : {res['prob_capital_loss']:.1%}")
print(f"  delta (+1% spot)          : {greeks['delta_1pct']:+.3f} pt")
print(f"  vega (+1 pt de vol)       : {greeks['vega_1pt']:+.3f} pt")

# --- Autocall : coupon fair vs volatilité ---
vols = np.linspace(0.15, 0.45, 7)
ac_coupons = [solve_fair_coupon(Autocall(S0, 2.0, v, r_d, r_f, n_obs=8, protection=0.60), TARGET, cfg_ac)
              for v in vols]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.plot(barriers * 100, np.array(brc_coupons) * 100, "o-", color="#1f77b4")
ax.set_xlabel("Barrière (% du niveau initial)")
ax.set_ylabel("Coupon annuel fair (%)")
ax.set_title(f"BRC 1 an : coupon fair vs barrière\n(prix cible {TARGET:.0f}%, vol {sigma:.0%})")
ax.grid(alpha=0.3)

ax = axes[1]
labels = [f"T{i+1}" for i in range(ac.n_obs)] + ["Maturité\n(non rappelé)"]
probs = list(res["autocall_probabilities"]) + [1 - res["autocall_probabilities"].sum()]
ax.bar(labels, np.array(probs) * 100, color=["#2ca02c"] * ac.n_obs + ["#d62728"])
ax.set_ylabel("Probabilité (%)")
ax.set_title(f"Autocall 2 ans : probabilité de rappel par date\n(coupon fair {ac.coupon:.2%} / trimestre)")
ax.grid(alpha=0.3, axis="y")

ax = axes[2]
ax.plot(vols * 100, np.array(ac_coupons) * 100, "o-", color="#ff7f0e")
ax.set_xlabel("Volatilité (%)")
ax.set_ylabel("Coupon fair par trimestre (%)")
ax.set_title("Autocall 2 ans : coupon fair vs volatilité\n(plus de vol = plus de coupon pour l'investisseur)")
ax.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "outputs", "07_structured_products.png")
plt.savefig(out, dpi=130)
print(f"Graphique sauvegardé : {out}")
