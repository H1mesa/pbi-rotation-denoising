# IA pour le débruitage de matrices de rotation (PBI T-EC-01)

Débruitage d'orientations mesurées par une centrale inertielle (IMU de drone ou de robot) à l'aide d'un réseau de neurones, en garantissant que les sorties restent de vraies rotations (groupe SO(3)). Le projet compare un réseau qui travaille dans l'espace tangent (`Log` en entrée, `Exp` en sortie) à un réseau qui optimise directement sur SO(3).

Projet Bachelor IPSA Toulouse, 3e année. Tuteur : Samy Labsir.

## Membres de l'équipe

- Hamesa Belgherras
- Axel Heuclin
- Arthur Ritz
- Jean Baptiste Truffier

## Installation / Utilisation

### Lancer les tests avec Docker (aucune autre installation)

```
docker build -t pbi-rotation-tests .
docker run --rm pbi-rotation-tests
```

### Sans Docker

```
pip install -r requirements.txt
python -m pytest -v
```

### Données

Téléchargez le jeu Auto MPG (UCI) et placez `auto-mpg.data` dans `data/` (ce fichier n'est pas versionné) :
https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data

### Étape A : régression linéaire et non linéaire

```
python -m experiments.etape_A_regression
```

Le script nettoie les données, entraîne les deux régressions par descente de gradient écrite à la main (split 80/20, graine 42), écrit les résultats dans `results/etape_A_regression.csv` et les figures dans `report/figures/`.

## Organisation

| Dossier | Contenu |
|---|---|
| `src/so3.py` | Outils SO(3) : `hat`, `vee`, `exp_map`, `log_map`, `geodesic_distance`, `is_rotation`, `project_to_so3`, `random_rotation`, `add_noise` |
| `src/regression.py` | Régressions à la main : modèles linéaire et exponentiel, descente de gradient, gradient check |
| `src/auto_mpg.py` | Chargement et nettoyage du jeu réel Auto MPG |
| `experiments/` | Scripts d'expérience |
| `tests/` | Tests unitaires (propriétés mathématiques, comparaison avec SciPy) |
| `results/` | Résultats chiffrés (CSV) |
| `report/figures/` | Figures pour les rapports |
| `docs/` | Comptes rendus d'étapes |
| `data/` | Données brutes (non versionnées) |

## Avancement

- [x] A : régression linéaire puis non linéaire en numpy ([compte rendu](docs/etape_A_regression.md))
- [x] B : utilitaires SO(3) testés
- [ ] C : données synthétiques + EuRoC
- [ ] D : moyenne de Fréchet par descente de gradient sur SO(3)
- [ ] E à G : réseau tangent, réseau recodé de zéro, réseau géométrique, comparaison
- [ ] H : données réelles et déploiement
