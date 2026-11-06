"""
Exemple 6 : Smile de volatilité FX et impact sur le pricing par rapport
à l'hypothèse de vol plate (Garman-Kohlhagen standard).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from src.vol_smile import price_with_smile

S0, T, r_d, r_f = 1.10, 0.5, 0.045, 0.030
sigma_atm, skew, curvature = 0.10, -0.025, 0.35  # skew négatif = smile typique EUR/USD (put skew)
strikes = np.linspace(0.90, 1.30, 41) * S0

df = price_with_smile(S0, T, r_d, r_f, strikes, sigma_atm, skew, curvature, "call")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.plot(df["moneyness_K_over_F"], df["smile_vol"] * 100, "o-", color="#1f77b4", ms=4)
ax.axhline(sigma_atm * 100, color="gray", ls="--", lw=1, label="Vol ATM (flat)")
ax.axvline(1.0, color="gray", ls=":", lw=1)
ax.set_xlabel("Moneyness (K / Forward)")
ax.set_ylabel("Volatilité implicite (%)")
ax.set_title("Smile de volatilité FX\n(paramétrisation quadratique)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
rel_diff_pct = (df["price_smile"] - df["price_flat_atm_vol"]) / df["price_flat_atm_vol"].clip(lower=1e-6) * 100
ax.plot(df["strike"], rel_diff_pct, "o-", color="#9467bd", ms=4)
ax.axhline(0, color="gray", ls="--", lw=1)
ax.axvline(S0, color="gray", ls=":", lw=1, label="Spot / ATM")
ax.set_xlabel("Strike")
ax.set_ylabel("Écart de prix (%) : smile vs vol plate")
ax.set_title("Sous/sur-évaluation du modèle flat-vol\npar strike (en % du prix réel avec smile)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(df["strike"], df["delta_smile"], "o-", label="Delta avec smile", color="#2ca02c", ms=4)
ax.plot(df["strike"], df["delta_flat_atm_vol"], "o-", label="Delta vol plate", color="#d62728", ms=4)
ax.set_xlabel("Strike")
ax.set_ylabel("Delta")
ax.set_title("Impact du smile sur le Delta\n(risque de mauvaise couverture si on ignore le smile)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("Volatility Smile FX — Impact sur Pricing & Delta vs Modèle Flat-Vol", fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "outputs", "06_vol_smile.png")
plt.savefig(out, dpi=140)
print(f"Graphe sauvegardé : {out}")

max_diff_pct = (df["price_diff"].abs() / df["price_flat_atm_vol"]).max() * 100
print(f"Écart de prix max (smile vs flat) : {max_diff_pct:.1f}% du prix flat-vol")
