"""
Étape A : régression linéaire puis non linéaire, en numpy, sur données réelles (Auto MPG).

Question : prédire la consommation (mpg) d'une voiture à partir de sa puissance (horsepower).
Lancer depuis le dossier du projet :  python -m experiments.etape_A_regression

Produit :
- results/etape_A_regression.csv          : un ligne par méthode (paramètres, MSE, RMSE, R²)
- report/figures/A1_donnees_et_modeles.png : nuage de points + les deux modèles
- report/figures/A2_courbes_de_perte.png   : perte train/test au fil des itérations
- report/figures/A3_pas_et_initialisation.png : effet du pas et du point de départ (non linéaire)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # pas de fenêtre : on écrit directement des fichiers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.auto_mpg import load_xy
from src.regression import (LinearModel, ExpDecayModel, Standardizer, gradient_descent,
                            train_test_split, mse, r2_score)

SEED = 42             # graine du découpage 80/20 (documentée, à garder pour le rapport)
LR = 0.1              # pas de la descente de gradient
N_ITER = 3000         # nombre d'itérations
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. Données réelles nettoyées, découpage 80/20
    x, y, rapport = load_xy()
    print("Nettoyage :", rapport)
    tr, te = train_test_split(len(x), 0.8, rng=SEED)

    # 2. Mise à l'échelle de x ET de y, statistiques calculées sur le train seulement
    sx, sy = Standardizer().fit(x[tr]), Standardizer().fit(y[tr])
    x_tr, x_te = sx.transform(x[tr]), sx.transform(x[te])
    y_tr, y_te = sy.transform(y[tr]), sy.transform(y[te])
    to_mpg = lambda z: sy.mean + sy.std * z            # repasser en unités réelles
    mse_scale = sy.std ** 2                            # MSE standardisée -> MSE en mpg²

    # 3. Entraînement des deux modèles avec le MÊME optimiseur
    modeles = {
        "lineaire": (LinearModel(), [0.0, 0.0]),
        "exponentielle": (ExpDecayModel(), [1.0, 0.5, 0.0]),
    }
    lignes, fits = [], {}
    for nom, (model, theta0) in modeles.items():
        theta, hist = gradient_descent(model, theta0, x_tr, y_tr, lr=LR, n_iter=N_ITER,
                                       x_test=x_te, y_test=y_te)
        fits[nom] = (model, theta, hist)
        pred_tr, pred_te = to_mpg(model.predict(theta, x_tr)), to_mpg(model.predict(theta, x_te))
        lignes.append({
            "methode": nom,
            "parametres_standardises": dict(zip(model.param_names, np.round(theta, 4))),
            "mse_train": mse(pred_tr, y[tr]),
            "mse_test": mse(pred_te, y[te]),
            "rmse_test_mpg": np.sqrt(mse(pred_te, y[te])),
            "r2_test": r2_score(pred_te, y[te]),
        })

    # Vérification : la descente linéaire doit retrouver la solution exacte des moindres carrés
    exact = LinearModel().closed_form(x_tr, y_tr)
    print("Linéaire, descente vs solution exacte :", fits["lineaire"][1].round(6), exact.round(6))

    resultats = pd.DataFrame(lignes)
    resultats.to_csv(RESULTS / "etape_A_regression.csv", index=False)
    print(resultats.round(3).to_string(index=False))

    # 4. Figure 1 : données et modèles, en unités réelles
    grille = np.linspace(x.min(), x.max(), 300)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x[tr], y[tr], s=12, alpha=0.5, label="entraînement (80 %)")
    ax.scatter(x[te], y[te], s=14, alpha=0.8, marker="s", label="test (20 %)")
    for nom, (model, theta, _) in fits.items():
        ax.plot(grille, to_mpg(model.predict(theta, sx.transform(grille))), lw=2.5, label=f"modèle {nom}")
    ax.set_xlabel("puissance (horsepower)")
    ax.set_ylabel("consommation (miles par gallon)")
    ax.set_title("Auto MPG : régression linéaire vs non linéaire")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "A1_donnees_et_modeles.png", dpi=150)

    # 5. Figure 2 : courbes de perte (train en trait plein, test en pointillés)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nom, (_, _, hist) in fits.items():
        line, = ax.plot(np.array(hist["train"]) * mse_scale, label=f"{nom} : train")
        ax.plot(np.array(hist["test"]) * mse_scale, ls="--", color=line.get_color(), label=f"{nom} : test")
    ax.set_yscale("log")
    ax.set_xlabel("itération de la descente de gradient")
    ax.set_ylabel("MSE (mpg²), échelle log")
    ax.set_title(f"Courbes de perte (pas = {LR})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "A2_courbes_de_perte.png", dpi=150)

    # 6. Figure 3 : sensibilité du modèle non linéaire au pas et à l'initialisation
    essais = [
        ("pas 0.01, départ (1, 0.5, 0)", 0.01, [1.0, 0.5, 0.0]),
        ("pas 0.1, départ (1, 0.5, 0)", 0.1, [1.0, 0.5, 0.0]),
        ("pas 0.3, départ (1, 0.5, 0)", 0.3, [1.0, 0.5, 0.0]),
        ("pas 0.1, départ (0, 0, 0)", 0.1, [0.0, 0.0, 0.0]),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, lr, theta0 in essais:
        _, hist = gradient_descent(ExpDecayModel(), theta0, x_tr, y_tr, lr=lr, n_iter=N_ITER)
        ax.plot(np.array(hist["train"]) * mse_scale, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("itération")
    ax.set_ylabel("MSE d'entraînement (mpg²), échelle log")
    ax.set_title("Non linéaire : le pas et le point de départ comptent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_pas_et_initialisation.png", dpi=150)

    print("Figures écrites dans", FIGURES)


if __name__ == "__main__":
    main()
