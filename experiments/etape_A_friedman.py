"""
Étape A (suite) : régressions écrites à la main sur le jeu make_friedman1 proposé par l'encadrant.

Données : 1000 points, 10 variables x1..x10 tirées uniformément dans [0, 1], et
    y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + bruit N(0, 1).
x6..x10 n'interviennent PAS dans y : ce sont des variables inutiles, un bon modèle doit les ignorer.
scikit-learn sert UNIQUEMENT à générer les données ; l'apprentissage est écrit à la main.

Lancer depuis le dossier du projet :  python -m experiments.etape_A_friedman

Produit :
- results/etape_A_friedman.csv            : une ligne par méthode (MSE, R², paramètres)
- report/figures/A4_friedman_pertes.png    : courbes de perte
- report/figures/A5_friedman_parametres.png : poids linéaires et paramètres retrouvés vs vrais
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1

from src.regression import (LinearMultiModel, FriedmanModel, gradient_descent,
                            train_test_split, mse, r2_score)

SEED_DONNEES = 0       # random_state donné par l'encadrant
SEED_SPLIT = 42        # graine du découpage 80/20
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
VRAIS = np.array([10.0, np.pi, 20.0, 0.5, 10.0, 5.0, 0.0])    # a, k, c, d, e, f, g


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    X, y = make_friedman1(n_samples=1000, n_features=10, noise=1.0, random_state=SEED_DONNEES)
    tr, te = train_test_split(len(y), 0.8, rng=SEED_SPLIT)
    # Pas de standardisation ici : les x sont déjà tous dans [0, 1], à la même échelle.

    essais = [
        # (nom, modèle, point de départ, pas, itérations)
        ("lineaire_10_variables", LinearMultiModel(10), np.zeros(11), 0.1, 5000),
        ("friedman_bon_depart", FriedmanModel(), [5, 2, 5, 0.3, 5, 5, 0], 0.005, 60000),
        ("friedman_depart_naif", FriedmanModel(), [1, 1, 1, 1, 1, 1, 0], 0.005, 60000),
    ]
    lignes, fits = [], {}
    for nom, model, theta0, lr, n_iter in essais:
        print(f"Entraînement : {nom} ({n_iter} itérations)...")
        theta, hist = gradient_descent(model, theta0, X[tr], y[tr], lr=lr, n_iter=n_iter,
                                       x_test=X[te], y_test=y[te])
        fits[nom] = (model, theta, hist)
        pred = model.predict(theta, X[te])
        lignes.append({
            "methode": nom,
            "pas": lr,
            "iterations": n_iter,
            "mse_train": hist["train"][-1],
            "mse_test": mse(pred, y[te]),
            "r2_test": r2_score(pred, y[te]),
            "parametres": dict(zip(model.param_names, np.round(theta, 3))),
        })

    lin_model, lin_theta, _ = fits["lineaire_10_variables"]
    print("Linéaire, descente vs solution exacte : écart max =",
          np.abs(lin_theta - lin_model.closed_form(X[tr], y[tr])).max())
    print("Référence : variance du bruit injecté = 1.0 (MSE minimale atteignable)")

    resultats = pd.DataFrame(lignes)
    resultats.to_csv(RESULTS / "etape_A_friedman.csv", index=False)
    print(resultats.drop(columns="parametres").round(3).to_string(index=False))

    # Figure 4 : courbes de perte
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nom, (_, _, hist) in fits.items():
        line, = ax.plot(hist["train"], label=f"{nom} : train")
        ax.plot(hist["test"], ls="--", color=line.get_color(), label=f"{nom} : test")
    ax.axhline(1.0, color="gray", lw=1, ls=":", label="variance du bruit (plancher)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("itération (échelle log)")
    ax.set_ylabel("MSE, échelle log")
    ax.set_title("make_friedman1 : courbes de perte")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_friedman_pertes.png", dpi=150)

    # Figure 5 : (gauche) poids linéaires, (droite) paramètres retrouvés vs vrais
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar([f"x{i}" for i in range(1, 11)], lin_theta[:-1])
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_title("Régression linéaire : poids appris")
    ax1.set_ylabel("poids w_i")
    ax1.set_xlabel("x6 à x10 n'influencent pas y : poids proches de 0")

    noms = FriedmanModel.param_names
    pos = np.arange(len(noms))
    ax2.bar(pos - 0.27, VRAIS, 0.27, label="vraie valeur")
    ax2.bar(pos, fits["friedman_bon_depart"][1], 0.27, label="bon départ")
    ax2.bar(pos + 0.27, fits["friedman_depart_naif"][1], 0.27, label="départ naïf")
    ax2.set_xticks(pos, noms)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_title("Modèle de Friedman : paramètres retrouvés")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "A5_friedman_parametres.png", dpi=150)

    print("Figures écrites dans", FIGURES)


if __name__ == "__main__":
    main()
