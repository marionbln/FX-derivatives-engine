"""
Exemple 5 : Surfaces de sensibilité (spot x vol) et décroissance temporelle (theta decay).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from src.sensitivity import price_surface_spot_vol, maturity_decay

K, T, r_d, r_f = 1.10, 0.5, 0.045, 0.030
spot_range, vol_range, Z = price_surface_spot_vol(K, T, r_d, r_f, "call")

decay_atm = maturity_decay(1.10, K, 0.10, r_d, r_f, "call")
decay_itm = maturity_decay(1.20, K, 0.10, r_d, r_f, "call")
decay_otm = maturity_decay(1.00, K, 0.10, r_d, r_f, "call")

fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
X, Y = np.meshgrid(spot_range, vol_range)
surf = ax1.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
ax1.set_xlabel("Spot")
ax1.set_ylabel("Volatilité")
ax1.set_zlabel("Prix")
ax1.set_title("Surface de prix Call FX\nen fonction du Spot et de la Vol (T=6M)")
fig.colorbar(surf, ax=ax1, shrink=0.6, label="Prix")

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(decay_atm.index * 365, decay_atm["price"], label="ATM (S=1.10)", color="#1f77b4", lw=2)
ax2.plot(decay_itm.index * 365, decay_itm["price"], label="ITM (S=1.20)", color="#2ca02c", lw=2)
ax2.plot(decay_otm.index * 365, decay_otm["price"], label="OTM (S=1.00)", color="#d62728", lw=2)
ax2.set_xlabel("Jours restants avant maturité")
ax2.set_ylabel("Prix de l'option")
ax2.set_title("Décroissance temporelle (time decay)\nCall K=1.10, vol=10%")
ax2.invert_xaxis()
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

plt.suptitle("Sensitivity Analysis — Prix vs (Spot, Vol, Maturité)", fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "outputs", "05_sensitivity.png")
plt.savefig(out, dpi=140)
print(f"Graphe sauvegardé : {out}")
