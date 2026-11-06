"""
Exemple 2 : Convergence du prix Monte Carlo vers le prix analytique GK,
et effet de la réduction de variance par variables antithétiques.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from src.garman_kohlhagen import FXOption
from src.monte_carlo import mc_price_vanilla, MCConfig

S0, K, T, sigma, r_d, r_f = 1.10, 1.08, 1.0, 0.12, 0.045, 0.030
analytical = FXOption(S0, K, T, sigma, r_d, r_f, "call").price()
print(f"Prix analytique GK : {analytical:.6f}")

path_counts = np.unique(np.geomspace(200, 200_000, 25).astype(int))

results_anti, results_std = [], []
for n in path_counts:
    r1 = mc_price_vanilla(S0, K, T, sigma, r_d, r_f, "call", MCConfig(n_paths=int(n), n_steps=1, antithetic=True, seed=1))
    r2 = mc_price_vanilla(S0, K, T, sigma, r_d, r_f, "call", MCConfig(n_paths=int(n), n_steps=1, antithetic=False, seed=1))
    results_anti.append(r1)
    results_std.append(r2)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(path_counts, [r["price"] for r in results_std], "o-", ms=3, label="MC standard", color="#d62728")
ax.plot(path_counts, [r["price"] for r in results_anti], "o-", ms=3, label="MC antithétique", color="#1f77b4")
ax.axhline(analytical, color="black", ls="--", lw=1.5, label="Prix analytique GK")
ax.set_xscale("log")
ax.set_xlabel("Nombre de trajectoires simulées")
ax.set_ylabel("Prix estimé")
ax.set_title("Convergence du prix Monte Carlo")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(path_counts, [r["stderr"] for r in results_std], "o-", ms=3, label="Erreur standard — MC standard", color="#d62728")
ax.plot(path_counts, [r["stderr"] for r in results_anti], "o-", ms=3, label="Erreur standard — antithétique", color="#1f77b4")
ax.plot(path_counts, results_std[0]["stderr"] / np.sqrt(path_counts / path_counts[0]), "k--", lw=1, label="Référence ~ 1/sqrt(N)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Nombre de trajectoires simulées")
ax.set_ylabel("Erreur standard")
ax.set_title("Réduction de variance (antithetic variates)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("Monte Carlo Pricing — Convergence & Réduction de Variance", fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "outputs", "02_mc_convergence.png")
plt.savefig(out, dpi=140)
print(f"Graphe sauvegardé : {out}")

final_anti = results_anti[-1]
print(f"MC final (N={final_anti['n_paths']}) : {final_anti['price']:.6f} "
      f"[{final_anti['ci_95'][0]:.6f}, {final_anti['ci_95'][1]:.6f}] "
      f"vs analytique {analytical:.6f}")
