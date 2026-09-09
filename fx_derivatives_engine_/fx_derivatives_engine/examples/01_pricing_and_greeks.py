"""
Exemple 1 : Pricing Garman-Kohlhagen et visualisation des Greeks EUR/USD.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from src.garman_kohlhagen import FXOption
from src.sensitivity import sensitivity_grid

# Paramètres marché EUR/USD indicatifs
S0, K, T, sigma, r_d, r_f = 1.10, 1.10, 0.5, 0.10, 0.045, 0.030

opt_call = FXOption(S0, K, T, sigma, r_d, r_f, "call")
opt_put = FXOption(S0, K, T, sigma, r_d, r_f, "put")
print("=== Pricing ATM EUR/USD, T=6M, vol=10% ===")
print(f"Call : {opt_call.price():.5f}   | Greeks : {opt_call.greeks()}")
print(f"Put  : {opt_put.price():.5f}   | Greeks : {opt_put.greeks()}")

# Parité put-call FX : Call - Put = S*exp(-r_f*T) - K*exp(-r_d*T)
lhs = opt_call.price() - opt_put.price()
rhs = S0 * np.exp(-r_f * T) - K * np.exp(-r_d * T)
print(f"Parité put-call : {lhs:.6f} vs {rhs:.6f} (écart={abs(lhs-rhs):.2e})")

grid_call = sensitivity_grid(K, T, sigma, r_d, r_f, "call")
grid_put = sensitivity_grid(K, T, sigma, r_d, r_f, "put")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
metrics = ["price", "delta", "gamma", "vega", "theta", "rho_domestic"]
titles = ["Prix", "Delta", "Gamma", "Vega (par 1% vol)", "Theta (par jour)", "Rho domestique (par 1%)"]

for ax, metric, title in zip(axes.flat, metrics, titles):
    ax.plot(grid_call.index, grid_call[metric], label="Call", color="#1f77b4", lw=2)
    ax.plot(grid_put.index, grid_put[metric], label="Put", color="#d62728", lw=2)
    ax.axvline(K, color="gray", ls="--", lw=1, alpha=0.6, label="Strike (ATM)")
    ax.set_title(title)
    ax.set_xlabel("Spot EUR/USD")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle("FX Vanilla Option — Prix & Greeks vs Spot (Garman-Kohlhagen)", fontsize=14, fontweight="bold")
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "outputs", "01_pricing_greeks.png")
plt.savefig(out, dpi=140)
print(f"Graphe sauvegardé : {out}")
