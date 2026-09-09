# FX Derivatives Pricing & Hedging Engine

Projet Python de pricing et de gestion du risque sur options de change (FX).
Le but : partir d'un pricer simple et construire un outil complet qui calcule
un prix, mesure le risque, gère des produits plus complexes, simule la
couverture de ce risque, et prend en compte la volatilité réelle du marché.

## Ce que fait le projet

**Pricing.** Calcul du prix d'une option de change avec le modèle
Garman-Kohlhagen (l'équivalent de Black-Scholes pour le FX, qui prend en
compte deux taux d'intérêt : le taux domestique et le taux étranger).
Le prix s'accompagne des Greeks : delta, gamma, vega, theta, rho — les
indicateurs qui disent comment le prix bouge si le marché bouge.

**Monte Carlo.** Le même prix recalculé autrement, en simulant des milliers
de trajectoires possibles du marché et en faisant la moyenne des résultats.
Sert à vérifier que le premier calcul est correct, et à pricer les produits
qui n'ont pas de formule simple.

**Options exotiques.** Deux produits plus complexes qu'une option classique :
les options à barrière (annulées ou activées si le prix touche un certain
niveau) et les options asiatiques (basées sur une moyenne du prix plutôt
que sur le prix final). Pricées par Monte Carlo, faute de formule directe.

**Couverture (hedging).** Simulation d'une banque qui vend une option et
ajuste sa position en continu pour se protéger des mouvements de marché.
Le projet montre l'effet de la fréquence d'ajustement sur le risque
résiduel, et l'effet d'un écart entre volatilité prévue et volatilité
réellement observée.

**Sensibilité.** Des graphiques montrant comment le prix évolue selon le
spot, la volatilité et le temps restant avant l'échéance.

**Smile de volatilité.** En pratique, la volatilité n'est pas la même pour
toutes les options : elle varie selon le prix d'exercice. Le projet montre
l'erreur commise si on l'ignore et qu'on utilise une seule volatilité pour
tout.

## Structure

```
src/
    garman_kohlhagen.py     prix et Greeks
    implied_volatility.py   retrouve la volatilité à partir d'un prix observé
    monte_carlo.py          simulation de trajectoires de marché
    exotics.py               options à barrière et asiatiques
    hedging.py               simulation de couverture et calcul du P&L
    sensitivity.py           graphiques de sensibilité
    vol_smile.py             smile de volatilité

examples/                  un script par partie, qui génère les graphiques
outputs/                   les graphiques générés
tests/                     tests de vérification
```

## Installation

```
pip install -r requirements.txt
python examples/01_pricing_and_greeks.py
```

Chaque script dans `examples/` correspond à une partie du projet et
produit un graphique dans `outputs/`.

## Vérifications

Le projet inclut des tests qui vérifient que les résultats sont cohérents :
le prix d'un call moins celui d'un put doit respecter une relation connue
(parité put-call), le prix obtenu par Monte Carlo doit être proche du prix
obtenu par la formule, et les options à barrière doivent se recomposer
correctement pour redonner une option classique. Tous ces tests passent.

```
python tests/test_pricing.py
```

## Limites

Les options exotiques sont pricées uniquement par simulation, ce qui est
plus lent qu'une formule fermée. Le smile de volatilité est modélisé de
façon simplifiée plutôt qu'avec un modèle de marché standard (SABR, par
exemple). La simulation de couverture ne prend pas en compte les coûts de
transaction. Ce sont des pistes d'amélioration possibles plutôt que des
manques bloquants pour l'usage démonstratif du projet.

## Outils utilisés

Python, NumPy, SciPy, Pandas, Matplotlib.
