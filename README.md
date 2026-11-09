# PBI T-EC-01 : IA pour le débruitage de matrices de rotation

Débruiter des matrices de rotation (IMU, robotique) avec un réseau de neurones, en garantissant
que les sorties restent dans SO(3). Contexte complet : [CLAUDE.md](CLAUDE.md).

## Installation

```bash
pip install -r requirements.txt
```

## Lancer les tests

```bash
python -m pytest -v
```

## Organisation

| Dossier | Contenu |
|---|---|
| `src/so3.py` | Outils SO(3) : `hat`, `vee`, `exp_map`, `log_map`, `geodesic_distance`, `is_rotation`, `project_to_so3`, `random_rotation`, `add_noise` |
| `tests/` | Tests unitaires (propriétés mathématiques, comparaison avec SciPy) |
| `data/` | Données brutes (non versionnées) |
| `notebooks/` | Expériences et figures |
| `report/` | Rapport intermédiaire et final |

## Avancement

- [x] B : utilitaires SO(3) testés
- [ ] A : régression linéaire puis non linéaire en numpy (pour le 27 novembre)
- [ ] C : données synthétiques + EuRoC
- [ ] D : moyenne de Fréchet par descente de gradient sur SO(3)
