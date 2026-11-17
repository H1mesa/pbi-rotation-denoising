# Jeux de données de capteurs mesurant des rotations

Projet : IA pour le débruitage de matrices de rotation (PBI T-EC-01)
Date : 17 novembre 2026

## Ce qu'il nous faut

Pour l'apprentissage supervisé, chaque instant doit fournir une **paire** :
- une mesure de capteur (IMU : gyromètre, accéléromètre, parfois magnétomètre) ou une orientation estimée bruitée ;
- une **vérité terrain** d'orientation, beaucoup plus précise, en général issue d'une capture de mouvement optique (caméras infrarouges + marqueurs : Vicon, Optitrack).

À défaut, on peut prendre des orientations de vérité terrain et les bruiter nous-mêmes avec `R · Exp(eps)` (décision de la réunion du 30 octobre).

## Comparatif

| Jeu | Support | Vérité terrain | Taille | Format | Intérêt pour nous |
|---|---|---|---|---|---|
| **BROAD** | IMU seule, mouvements variés | Capture optique synchronisée | 39 essais (23 sans perturbation, 16 avec perturbations magnétiques et d'accélération) | Matlab `.mat`, HDF5 | **Le plus proche du sujet** : conçu pour évaluer l'estimation d'orientation. Licence CC BY 4.0, code Python fourni |
| **EuRoC MAV** | Drone (IMU 200 Hz + caméras) | Vicon / Leica | 11 séquences | CSV (format ASL) | Référence en robotique, citée dans les articles du domaine. Quaternion écrit `w, x, y, z` |
| **TUM VI** | Caméra + IMU portées à la main | Capture de mouvement sur une partie des séquences | 28 séquences | CSV | Alternative à EuRoC |
| **OxIOD** | Smartphone (main, poche, sac, chariot) | Vicon sur 132 séquences sur 158 | 42 km au total | CSV, quaternion `x, y, z, w` | Beaucoup de données, mais orientation de vérité terrain signalée comme irrégulière par endroits |

## Recommandation

1. **Commencer par BROAD** : petit, propre, orienté exactement vers l'estimation d'orientation, avec une vérité terrain optique et des essais perturbés qui serviront de « capteur très bruité ».
2. **Puis EuRoC MAV** (séquence `MH_01_easy`) pour un cas drone, plus proche du discours de la présentation de Samy Labsir.
3. Garder les données de l'ENAC (annoncées par l'encadrant) pour la validation finale.

## Premier résultat sur BROAD (17 novembre 2026)

Téléchargé dans `data/broad/` (39 essais, 159 Mo, 4,2 % d'orientations de référence manquantes en moyenne, retirées).
La régression linéaire d'Arthur apprend une correction du gyromètre `omega ~ gyro W + B` (31 essais d'entraînement, 8 de test) :

- W est proche de l'identité (erreur d'échelle 0,5 %, couplage d'axes jusqu'à 1,6 %), biais moyen d'environ 0,0015 rad/s ;
- sur la vitesse instantanée, le gain est négligeable (0,2032 -> 0,2031 rad/s) : le bruit domine ;
- sur l'orientation obtenue en intégrant le gyromètre seul pendant 3 minutes, l'erreur moyenne passe de 33,4° à 28,0°, avec 7 essais de test améliorés sur 8 ;
- l'essai 15 est dégradé : le biais change d'un essai à l'autre, une correction unique ne peut pas s'y adapter. C'est ce qui motive une correction qui s'adapte (filtre de Kalman, réseau).

Détails : `experiments/etape_A_broad.py`, `results/etape_A_broad.csv`, figures A6 et A7.

## Pièges à vérifier à chaque chargement

- **Ordre du quaternion** : EuRoC donne `w, x, y, z`, SciPy attend `x, y, z, w`. OxIOD donne déjà `x, y, z, w`.
- **Synchronisation** : l'IMU et la vérité terrain n'ont pas la même fréquence ; il faut aligner les horodatages.
- **Qualité de la vérité terrain** : la documentation d'OpenVINS signale une orientation de vérité terrain peu précise sur EuRoC `V1_01_easy`.
- Après conversion, toujours vérifier `R^T R ≈ I` et `det(R) ≈ 1` (`src.so3.is_rotation`).

## Sources

- BROAD : https://github.com/dlaidig/broad ; article : Laidig, Caruso, Cereatti, Seel, *BROAD — A Benchmark for Robust Inertial Orientation Estimation*, Data 6(7), 2021, https://www.mdpi.com/2306-5729/6/7/72
- EuRoC MAV : https://projects.asl.ethz.ch/datasets/euroc-mav/ (fichiers hébergés sur l'ETH Research Collection)
- TUM VI : https://cvg.cit.tum.de/data/datasets/vi-dataset
- OxIOD : http://deepio.cs.ox.ac.uk ; article : https://arxiv.org/abs/1809.07491
- Remarque sur V1_01_easy : https://docs.ros.org/en/api/ov_core/html/gs-datasets.html
