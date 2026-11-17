"""
Étape A sur données réelles de rotation : la régression linéaire d'Arthur appliquée au jeu BROAD.

Question : peut-on corriger le gyromètre d'une IMU avec une régression linéaire apprise sur des
données de vérité terrain (caméras Optitrack), et est-ce que ça réduit l'erreur d'orientation ?

Modèle : vraie vitesse de rotation  omega ~ gyro @ W + B   (W : 3x3, échelle et couplage des axes ; B : biais)
- les entrées/sorties sont construites par src.broad.angular_rate_pairs ;
- découpage 80/20 PAR ESSAI (31 essais d'entraînement, 8 de test, graine 42) : on ne mélange jamais
  des instants d'un même essai entre train et test, sinon le test serait trop facile ;
- évaluation sur les rotations : on intègre le gyromètre brut et le gyromètre corrigé
  (R_{k+1} = R_k Exp(omega dt)) et on mesure l'erreur géodésique, en degrés, par rapport aux caméras.

Lancer depuis le dossier du projet :  python -m experiments.etape_A_broad   (environ 1 minute)
Données : data/broad/*.mat (voir src/broad.py).

Produit :
- results/etape_A_broad.csv          : une ligne par essai de test (erreurs brute et corrigée)
- report/figures/A6_broad_derive.png : erreur d'orientation au cours du temps, un essai de test
- report/figures/A7_broad_essais.png : erreur moyenne par essai de test, brut vs corrigé
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.broad import list_trials, load_trial, angular_rate_pairs, integrate_gyro
from src.regression_arthur import fit, predict, solution_exacte, split_80_20
from src.so3 import geodesic_distance

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"


def orientation_error_deg(trial, gyr):
    """Intègre `gyr` depuis la première orientation de référence et renvoie (temps, erreur en degrés)."""
    k0 = int(np.argmax(trial["valid"]))                         # premier instant vu par les caméras
    R = integrate_gyro(gyr[k0:], trial["R_ref"][k0], trial["fs"])
    v = trial["valid"][k0:]
    t = np.arange(len(R))[v] / trial["fs"]
    return t, np.degrees(geodesic_distance(R[v], trial["R_ref"][k0:][v]))


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    trials = [load_trial(p) for p in list_trials()]
    if not trials:
        raise SystemExit("Données absentes : copiez data_mat/*.mat du dépôt BROAD dans data/broad/")
    manquants = np.mean([1 - t["valid"].mean() for t in trials])
    print(f"{len(trials)} essais, {100 * manquants:.1f} % d'orientations de référence manquantes en moyenne")

    tr, te = split_80_20(len(trials), seed=SEED)
    paires = [angular_rate_pairs(t) for t in trials]
    X_tr = np.vstack([paires[i][0] for i in tr]); Y_tr = np.vstack([paires[i][1] for i in tr])
    X_te = np.vstack([paires[i][0] for i in te]); Y_te = np.vstack([paires[i][1] for i in te])
    print(f"Paires d'apprentissage : {len(X_tr)} (train), {len(X_te)} (test)")

    # Régression d'Arthur : 3 entrées (gyro x, y, z) -> 3 sorties (vraie vitesse x, y, z)
    W, B, hist = fit(X_tr, Y_tr, learning_rate=0.1, nb_iter=500, X_test=X_te, Y_test=Y_te)
    W_ex, B_ex = solution_exacte(X_tr, Y_tr)
    print("Écart descente / solution exacte :", np.abs(W - W_ex).max())
    print("W appris (identité = gyromètre parfaitement calibré) :\n", W.round(4))
    print("B appris (biais, rad/s) :", B.round(5))

    rms = lambda E: float(np.sqrt(np.mean(np.sum(E ** 2, axis=1))))
    print(f"Erreur de vitesse sur le test : brute {rms(X_te - Y_te):.4f} rad/s, "
          f"corrigée {rms(predict(X_te, W, B) - Y_te):.4f} rad/s")

    # Évaluation sur les orientations des essais de test
    lignes, courbes = [], {}
    for i in te:
        t = trials[i]
        temps, err_brut = orientation_error_deg(t, t["gyr"])
        _, err_corr = orientation_error_deg(t, predict(t["gyr"], W, B))
        courbes[t["name"]] = (temps, err_brut, err_corr)
        lignes.append({
            "essai": t["name"],
            "duree_s": round(len(t["gyr"]) / t["fs"], 1),
            "erreur_moyenne_brute_deg": err_brut.mean(),
            "erreur_moyenne_corrigee_deg": err_corr.mean(),
            "erreur_finale_brute_deg": err_brut[-1],
            "erreur_finale_corrigee_deg": err_corr[-1],
        })
    res = pd.DataFrame(lignes)
    res.to_csv(RESULTS / "etape_A_broad.csv", index=False)
    print(res.round(2).to_string(index=False))
    ameliores = (res["erreur_moyenne_corrigee_deg"] < res["erreur_moyenne_brute_deg"]).sum()
    print(f"Essais améliorés : {ameliores}/{len(res)} ; erreur moyenne "
          f"{res['erreur_moyenne_brute_deg'].mean():.1f}° -> {res['erreur_moyenne_corrigee_deg'].mean():.1f}°")

    # Figure 6 : dérive au cours du temps pour le premier essai de test
    nom = res["essai"].iloc[0]
    temps, eb, ec = courbes[nom]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(temps, eb, label="gyromètre brut")
    ax.plot(temps, ec, label="gyromètre corrigé (régression)")
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("erreur d'orientation (degrés)")
    ax.set_title(f"BROAD, essai de test {nom}\nintégration du gyromètre seul : l'erreur s'accumule")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "A6_broad_derive.png", dpi=150)

    # Figure 7 : erreur moyenne par essai de test
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos = np.arange(len(res))
    ax.bar(pos - 0.2, res["erreur_moyenne_brute_deg"], 0.4, label="brut")
    ax.bar(pos + 0.2, res["erreur_moyenne_corrigee_deg"], 0.4, label="corrigé")
    ax.set_xticks(pos, [n[:2] for n in res["essai"]])
    ax.set_xlabel("numéro de l'essai de test")
    ax.set_ylabel("erreur d'orientation moyenne (degrés)")
    ax.set_title("BROAD : effet de la correction linéaire du gyromètre")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "A7_broad_essais.png", dpi=150)
    print("Figures écrites dans", FIGURES)


if __name__ == "__main__":
    main()
