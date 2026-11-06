"""
Exemple 4 : Simulation de couverture delta dynamique — distribution du P&L
et effet de la fréquence de rebalancement + effet vol implicite vs réalisée.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from src.hedging import simulate_delta_hedge, hedging_frequency_study

S0, K, T, sigma, r_d, r_f = 1.10, 1.10, 0.5, 0.10, 0.045, 0.030

# --- 1. Distribution du P&L pour un rebalancement quotidien ---
res_daily = simulate_delta_hedge(S0, K, T, sigma, r_d, r_f, "call", n_rebalances=126, n_paths=20000, seed=1)
res_weekly = simulate_delta_hedge(S0, K, T, sigma, r_d, r_f, "call", n_rebalances=26, n_paths=20000, seed=1)

print(f"Prime encaissée à t=0 : {res_daily.option_price0:.5f}")
print(f"P&L hedging (quotidien) : moyenne={res_daily.pnl.mean():.6f}, std={res_daily.pnl.std():.6f}")
print(f"P&L hedging (hebdo)     : moyenne={res_weekly.pnl.mean():.6f}, std={res_weekly.pnl.std():.6f}")

# --- 2. Effet de la fréquence de rebalancement sur la variance du P&L ---
freq_study = hedging_frequency_study(S0, K, T, sigma, r_d, r_f, "call",
                                       frequencies=(1, 4, 12, 26, 52, 126, 252), n_paths=15000, seed=1)

# --- 3. Effet vol implicite vs vol réalisée (mismatch) ---
realized_vols = np.linspace(0.05, 0.20, 15)
pnl_means, pnl_stds = [], []
for rv in realized_vols:
    r = simulate_delta_hedge(S0, K, T, sigma, r_d, r_f, "call", n_rebalances=126,
                               n_paths=15000, seed=3, realized_sigma=rv)
    pnl_means.append(r.pnl.mean()); pnl_stds.append(r.pnl.std())

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
ax.hist(res_daily.pnl, bins=60, alpha=0.6, label="Rebal. quotidien (126x)", color="#1f77b4", density=True)
ax.hist(res_weekly.pnl, bins=60, alpha=0.6, label="Rebal. hebdo (26x)", color="#d62728", density=True)
ax.axvline(0, color="black", ls="--", lw=1)
ax.set_xlabel("P&L de couverture à maturité")
ax.set_ylabel("Densité")
ax.set_title("Distribution du P&L de hedging\n(vol implicite = vol réalisée)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
freqs = list(freq_study.keys())
stds = [freq_study[f]["std_pnl"] for f in freqs]
ax.plot(freqs, stds, "o-", color="#2ca02c")
ax.set_xscale("log")
ax.set_xlabel("Nombre de rebalancements sur la vie de l'option")
ax.set_ylabel("Écart-type du P&L de couverture")
ax.set_title("Effet de la fréquence de hedging\nsur le risque résiduel (gamma risk)")
ax.grid(alpha=0.3)

ax = axes[2]
ax.plot(realized_vols, pnl_means, "o-", color="#ff7f0e")
ax.axvline(sigma, color="black", ls="--", lw=1, label="Vol implicite pricée")
ax.axhline(0, color="gray", ls=":", lw=1)
ax.set_xlabel("Volatilité réalisée du marché")
ax.set_ylabel("P&L moyen de hedging")
ax.set_title("P&L moyen vs vol réalisée\n(vendre vol chère = profit si vol réalisée < implicite)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.suptitle("Delta Hedging — Analyse du P&L de Couverture Dynamique", fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "outputs", "04_delta_hedging.png")
plt.savefig(out, dpi=140)
print(f"Graphe sauvegardé : {out}")
