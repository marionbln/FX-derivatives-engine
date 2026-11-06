"""
Exemple 3 : Pricing d'options exotiques FX — Barrier (Up-and-Out) & Asian.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from src.garman_kohlhagen import FXOption
from src.monte_carlo import MCConfig
from src.exotics import price_barrier_option, price_asian_option

S0, K, T, sigma, r_d, r_f = 1.10, 1.10, 0.5, 0.12, 0.045, 0.030
config = MCConfig(n_paths=200_000, n_steps=126, seed=7)

vanilla_call = FXOption(S0, K, T, sigma, r_d, r_f, "call").price()
print(f"Vanille (référence) : {vanilla_call:.5f}")

# --- Barrier : effet du niveau de barrière sur le prix Up-and-Out ---
barriers = np.linspace(1.11, 1.35, 20)
uo_prices, uo_stderr = [], []
for b in barriers:
    r = price_barrier_option(S0, K, b, T, sigma, r_d, r_f, "up", "out", "call", config)
    uo_prices.append(r["price"]); uo_stderr.append(r["stderr"])

# --- Asian : comparaison call asiatique vs call vanille pour différents strikes ---
strikes = np.linspace(0.95, 1.25, 20) * S0
asian_prices, vanilla_prices = [], []
for k in strikes:
    a = price_asian_option(S0, k, T, sigma, r_d, r_f, "call", config)
    v = FXOption(S0, k, T, sigma, r_d, r_f, "call").price()
    asian_prices.append(a["price"]); vanilla_prices.append(v)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(barriers, uo_prices, "o-", color="#1f77b4", ms=4)
ax.fill_between(barriers, np.array(uo_prices) - 1.96*np.array(uo_stderr),
                 np.array(uo_prices) + 1.96*np.array(uo_stderr), alpha=0.2, color="#1f77b4")
ax.axhline(vanilla_call, color="black", ls="--", lw=1.5, label="Call vanille (référence)")
ax.axvline(S0, color="gray", ls=":", lw=1, label="Spot actuel")
ax.set_xlabel("Niveau de barrière (Up)")
ax.set_ylabel("Prix")
ax.set_title("Call Up-and-Out : prix vs niveau de barrière\n(converge vers 0 quand barrière proche du spot, vers vanille quand barrière loin)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(strikes, vanilla_prices, "o-", label="Call vanille", color="#d62728", ms=4)
ax.plot(strikes, asian_prices, "o-", label="Call asiatique (moyenne arithm.)", color="#2ca02c", ms=4)
ax.set_xlabel("Strike")
ax.set_ylabel("Prix")
ax.set_title("Asiatique vs Vanille\n(l'asiatique est moins chère : la moyenne réduit la volatilité effective)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("Options Exotiques FX — Barrier & Asian (Monte Carlo)", fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "outputs", "03_exotics.png")
plt.savefig(out, dpi=140)
print(f"Graphe sauvegardé : {out}")
