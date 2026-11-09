# CLAUDE.md : PBI T-EC-01, IA pour le débruitage de matrices de rotation

> Ce fichier est le contexte complet du projet. Lis-le en entier avant d'écrire du code.

## 1. Qui, quoi, quand

- **Projet** : PBI (projet Bachelor IPSA Toulouse, 3e année), code **T-EC-01**, secteur IA, sujet exploratoire.
- **Équipe** : 3 à 4 étudiants de Bachelor 3 (dont Axel Heuclin). 200 h par étudiant, de septembre à avril.
- **Encadrement** : Hamza Belgherras (suivi des réunions) et M. Pen Daouda. Tuteur officiel du catalogue : Samy Labsir (IPSA, spécialiste du filtrage de Kalman sur groupes de Lie).
- **Notation** (à confirmer par l'équipe) : rapport, soutenance, et évaluation de l'implémentation par l'encadrement. Un rapport intermédiaire est à rendre fin décembre ou en janvier (il entre dans la note finale). Pénalités : jusqu'à 4 points pour un retard de livrable non justifié, 2 points pour un oubli de dépôt Moodle.
- **Prochain point formel** : vendredi 27 novembre 2026 au matin. Les partiels sont début décembre.

## 2. Niveau des utilisateurs : IMPORTANT

Les étudiants sont en 3e année de Bachelor. Ils ne maîtrisent pas encore l'algèbre de Lie. Donc :
- Explique ce que tu fais avant de le faire, avec des mots simples et une analogie si utile.
- Commente le code en français, de façon pédagogique (pourquoi, pas seulement quoi).
- Ne remplace pas leur travail par un code opaque : ils doivent **comprendre et pouvoir expliquer** chaque brique à l'oral. Propose, explique, puis implémente.
- Quand tu introduis un terme (SO(3), so(3), Exp, Log, géodésique, rétropropagation), rappelle sa définition en une phrase.

## 3. Le problème en une page

Un capteur d'orientation (IMU de drone ou de robot) donne des **matrices de rotation bruitées**. On veut les débruiter avec un réseau de neurones **en gardant des sorties qui sont de vraies rotations**.

**Pourquoi c'est difficile** : une matrice de rotation R appartient à **SO(3)**, c'est-à-dire R^T R = I et det(R) = +1. L'opération naturelle est la **multiplication** (non commutative), pas l'addition. Donc `R + bruit` n'est **pas** une rotation. Le bruit doit être **multiplicatif**.

### Outils mathématiques de base

- **SO(3)** : groupe des rotations 3D, variété courbe de dimension 3.
- **so(3)** : algèbre de Lie = matrices antisymétriques 3x3 ≃ vecteurs de R^3 (espace plat, tangent à SO(3) en l'identité). `hat` : vecteur → matrice antisymétrique, `vee` : l'inverse.
- **Exp : so(3) → SO(3)** (formule de Rodrigues) : `R = I + (sin θ/θ) [ω]x + ((1−cos θ)/θ²) [ω]x²`, avec `θ = ‖ω‖`.
- **Log : SO(3) → so(3)** : `θ = arccos((tr R − 1)/2)`, `[ω]x = θ/(2 sin θ) (R − R^T)`. Attention aux cas limites θ ≈ 0 et θ ≈ π (utiliser `scipy.spatial.transform.Rotation` plutôt que du code fait main, ou gérer ces cas).
- **Bruit correct** : `R_bruitee = R_vraie @ Exp(eps)` avec `eps ~ N(0, sigma² I3)` dans so(3). Jamais `R_vraie + bruit`.
- **Erreur géodésique** (métrique et fonction de perte) : `d(R1, R2) = ‖Log(R1^T R2)‖ = arccos((tr(R1^T R2) − 1)/2)`, en radians (convertir en degrés pour l'affichage).
- **Moyenne de Fréchet** : moyenne d'un ensemble de rotations par itération : `mu ← mu · Exp(moyenne_i Log(mu^T R_i))`.
- **Descente de gradient sur SO(3)** : calculer le gradient riemannien dans so(3), puis `R ← R · Exp(−η · grad)`. Ainsi R reste toujours dans SO(3).
- Il ne faut pas confondre : espace **tangent** (plat, fiable seulement localement) et **groupe** (courbe).

## 4. Décisions prises en réunion (1er oct. et 30 oct. 2026)

1. **Apprentissage supervisé** pour l'instant.
2. On commence par des **données synthétiques** (split 80 % entraînement / 20 % test), puis on passe à des **données réelles d'IMU** (plateforme robotique Alipsa ; données de trajectoire d'une centrale inertielle de l'ENAC annoncées par Hamza).
3. Pas de vérité terrain ? On la fabrique : soit du bruit artificiel à plusieurs niveaux de variance sur des rotations réelles, soit un débruitage par un modèle simple utilisé comme référence.
4. **Régression d'abord, sans bibliothèque** (numpy uniquement, gradient à la main). Une régression **non linéaire** est demandée (les optimisations itératives sont plus proches de ce qui vient ensuite).
5. Le réseau de neurones (2 à 3 couches) doit être **recodé de zéro** (pas de simple fine-tuning). Un réseau est une composition de fonctions : la régression est le cas à une fonction.
6. **Objectif de recherche** : faire de la **descente de gradient directement sur SO(3)** dans le réseau (la géométrie dans les couches), pas seulement travailler dans l'espace tangent (`Log` en entrée, couches classiques, `Exp` en sortie). Hypothèse de l'encadrant : l'approche tangente perd en précision quand le bruit est fort (espace courbe), donc **comparer les deux approches** (décision actée, à rediscuter).
7. À étudier : est-ce qu'on peut modifier la rétropropagation en profondeur dans PyTorch (autograd personnalisé, `torch.autograd.Function`) ? Pistes à vérifier : `geoopt`, `geomstats`.
8. Le filtre de Kalman sur groupes de Lie (Samy Labsir) est une **baseline à comprendre**, pas à coder en priorité.
9. Déploiement (FastAPI, Docker) : en fin de projet.

## 5. Données

### Données synthétiques
Générer N rotations vraies (aléatoires ou trajectoires lisses), puis les bruiter avec `R_vraie @ Exp(eps)`, pour plusieurs `sigma` (par exemple 0.01, 0.05, 0.1, 0.3, 0.5 rad). Split 80/20 avec graine fixe.

### Données réelles : EuRoC MAV (piste principale)
- Page : https://projects.asl.ethz.ch/datasets/euroc-mav/ (format ASL, CSV). Commencer par `V1_01_easy` ou `MH_01_easy`.
- `mav0/imu0/data.csv` : gyroscope et accéléromètre à 200 Hz.
- `mav0/state_groundtruth_estimate0/data.csv` : vérité terrain (Vicon), orientation en **quaternion**.
- **Piège** : EuRoC écrit le quaternion en `w, x, y, z`, SciPy attend `x, y, z, w`. Vérifier les en-têtes du CSV.
- Alternative : TUM VI (https://cvg.cit.tum.de/data/datasets/vi-dataset), vérité terrain par capture de mouvement seulement sur une partie des séquences.
- Nettoyage obligatoire : valeurs manquantes, doublons, alignement des horodatages IMU/vérité terrain.
- Toujours vérifier après conversion : `R^T R ≈ I` et `det ≈ 1`.

```python
import numpy as np
from scipy.spatial.transform import Rotation as Rot

q_scipy = q[:, [1, 2, 3, 0]]                    # (w,x,y,z) -> (x,y,z,w)
R_vraie = Rot.from_quat(q_scipy).as_matrix()    # (N, 3, 3)
eps = np.random.normal(0, sigma, (len(R_vraie), 3))
R_bruitee = R_vraie @ Rot.from_rotvec(eps).as_matrix()
err = np.linalg.norm(Rot.from_matrix(R_vraie.transpose(0,2,1) @ R_bruitee).as_rotvec(), axis=1)
```

## 6. Plan de travail proposé

| Étape | Contenu | Livrable |
|---|---|---|
| A (avant le 27 nov.) | Régression linéaire puis **non linéaire** en numpy sur données réelles trouvées en ligne (nettoyées), descente de gradient écrite à la main | Notebook ou script + courbes de perte |
| B | Utilitaires SO(3) : `hat`, `vee`, `Exp`, `Log`, erreur géodésique, tests unitaires (propriétés : `Exp(Log(R)) = R`, orthogonalité) | Module `so3.py` testé |
| C | Génération des données synthétiques + chargement EuRoC + split 80/20 | Module `data.py` |
| D | **Passerelle** : estimer une rotation inconnue R à partir de N mesures `R·Exp(eps_i)` par descente de gradient sur SO(3) (moyenne de Fréchet) | Notebook, comparaison avec la moyenne naïve des matrices |
| E | Baseline réseau : entrée `Log(R_bruitee)`, couches classiques, sortie `Exp` (version tangente) | Résultats par niveau de bruit |
| F | Réseau à 2 couches recodé de zéro, rétropropagation à la main | Réseau fonctionnel |
| G | Version géométrique : gradient sur SO(3) dans le réseau, et comparaison avec E selon `sigma` | Courbe erreur géodésique vs niveau de bruit |
| H | Données réelles (EuRoC, données ENAC), puis déploiement FastAPI + Docker | Démo |
| I | Rapport intermédiaire (déc./janv.), rapport final (dépôt Moodle) et soutenance (avril) | Documents |

Les étapes A à D sont la priorité immédiate. Ne pas sauter à F et G avant d'avoir B, C et D solides.

## 7. Structure de dépôt suggérée

```
pbi-rotation-denoising/
├── CLAUDE.md
├── README.md
├── requirements.txt          # numpy, scipy, matplotlib, pandas, torch (étapes E et G), pytest
├── data/                     # données brutes, non versionnées (voir .gitignore)
├── src/
│   ├── so3.py                # hat, vee, exp_map, log_map, geodesic_error, projection sur SO(3)
│   ├── data.py               # génération synthétique, chargement EuRoC, split
│   ├── regression.py         # régressions écrites à la main
│   ├── nn_scratch.py         # réseau recodé de zéro
│   └── nn_so3.py             # version géométrique
├── notebooks/
├── tests/
└── report/
```

## 8. Conventions et règles

- **Python 3.10 ou plus**, numpy vectorisé, tailles de tenseurs notées en commentaire (`# (N, 3, 3)`).
- Graine aléatoire fixe et documentée pour toute expérience.
- Toute fonction sur SO(3) a un test : orthogonalité, déterminant 1, `Exp(Log(R)) ≈ R`, `Log(Exp(w)) ≈ w` pour `‖w‖ < π`.
- Pas de bibliothèque de haut niveau pour la régression et le réseau de l'étape F (consigne de l'encadrant). SciPy sert pour `Rotation` et pour **vérifier** nos implémentations, pas pour les remplacer au début.
- Les figures : erreur géodésique en degrés, axes légendés, une courbe par méthode.
- Chaque expérience produit un fichier de résultats (CSV) réutilisable pour le rapport.
- Commits petits et messages clairs en français.

## 9. Points de vigilance

- Ne jamais additionner des matrices de rotation ni leur ajouter du bruit directement.
- Ne jamais moyenner des matrices telles quelles (utiliser la moyenne de Fréchet, ou une projection SVD de la moyenne en solution rapide).
- Stabilité numérique de `Log` près de θ = 0 et θ = π.
- Surapprentissage : toujours évaluer sur le jeu de test jamais vu.
- Une comparaison honnête : même données, mêmes niveaux de bruit, même métrique pour toutes les méthodes.

## 10. Références (vérifiées)

- Solà et al., *A micro Lie theory for state estimation in robotics* : https://arxiv.org/abs/1812.01537
- Zhou et al., *On the Continuity of Rotation Representations in Neural Networks* : https://arxiv.org/abs/1812.07035
- Brossard, Bonnabel, Barrau, *Denoising IMU Gyroscopes with Deep Learning for Open-Loop Attitude Estimation* : https://arxiv.org/abs/2002.10718
- Brossard et al., *AI-IMU Dead-Reckoning* (code) : https://github.com/mbrossar/ai-imu-dr
- Revach et al., *KalmanNet* : https://arxiv.org/abs/2107.10043
- Geomstats : https://github.com/geomstats/geomstats

## 11. Par où commencer maintenant

1. Propose un `requirements.txt` et la structure de dépôt ci-dessus, puis attends validation.
2. Écris et teste `src/so3.py` (étape B) en expliquant chaque fonction.
3. Guide l'équipe pas à pas sur la régression non linéaire de l'étape A, avec des données réelles à nettoyer.
4. À chaque étape, termine par : ce qui a été fait, ce qu'il faut comprendre avant de continuer, et la prochaine action.
