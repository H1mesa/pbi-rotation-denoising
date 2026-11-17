"""
broad.py : chargement et nettoyage du jeu réel BROAD (Berlin Robust Orientation Estimation Assessment Dataset).

Source : https://github.com/dlaidig/broad (licence CC BY 4.0), article : Laidig et al., Data 6(7), 2021.
Téléchargement : dossier data_mat/ du dépôt, à copier dans data/broad/ (non versionné).

Chaque essai (39 au total) contient, à 285,7 Hz :
- imu_gyr  (N, 3) : vitesse de rotation mesurée par le gyromètre, en rad/s (le capteur bruité) ;
- opt_quat (N, 4) : orientation de référence mesurée par caméras Optitrack (la vérité terrain),
                    quaternion unitaire dans l'ordre w, x, y, z (SciPy attend x, y, z, w !) ;
- movement (N,)   : masque des instants où le capteur bouge.

Défaut à nettoyer : quand un marqueur est caché aux caméras, opt_quat vaut NaN (0,3 % à 15 % des instants).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as Rot

from src.so3 import exp_map, log_map, is_rotation

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "broad"


def list_trials(directory=DEFAULT_DIR):
    """Chemins des 39 essais, triés par numéro."""
    return sorted(Path(directory).glob("[0-9][0-9]_*.mat"))


def load_trial(path):
    """Lit un essai et convertit les quaternions de référence en matrices de rotation.

    Renvoie un dict : gyr (N, 3), R_ref (N, 3, 3) (NaN là où les caméras ont perdu le marqueur),
    valid (N,) booléen, fs (Hz), movement (N,), name.
    """
    d = loadmat(path, squeeze_me=True)
    q = d["opt_quat"]                                           # (N, 4) en w, x, y, z
    valid = ~np.isnan(q).any(axis=1)                            # (N,)
    R_ref = np.full((len(q), 3, 3), np.nan)
    R_ref[valid] = Rot.from_quat(q[valid][:, [1, 2, 3, 0]]).as_matrix()   # -> x, y, z, w
    assert is_rotation(R_ref[valid], tol=1e-6).all(), "conversion quaternion -> rotation incorrecte"
    return {
        "name": Path(path).stem,
        "gyr": d["imu_gyr"].astype(float),
        "R_ref": R_ref,
        "valid": valid,
        "fs": float(d["sampling_rate"]),
        "movement": d["movement"].astype(bool),
    }


def angular_rate_pairs(trial, window=10):
    """Paires (gyromètre mesuré, vraie vitesse de rotation) pour l'apprentissage supervisé.

    La vraie vitesse de rotation se déduit de deux orientations de référence espacées de `window`
    échantillons :  omega = Log(R_k^T R_{k+window}) / (window * dt).
    C'est l'écart entre deux rotations (sur SO(3)), ramené dans l'espace plat so(3) par Log,
    divisé par le temps. On la compare au gyromètre MOYENNÉ sur la même fenêtre.
    Une fenêtre de 10 échantillons (35 ms) atténue le bruit de dérivation des caméras.

    On garde seulement les fenêtres entièrement couvertes par les caméras (pas de NaN).
    Renvoie X (M, 3) gyromètre, Y (M, 3) vraie vitesse, en rad/s.
    """
    valid, R, g, fs = trial["valid"], trial["R_ref"], trial["gyr"], trial["fs"]
    n = len(g)
    starts = np.arange(0, n - window, window)                   # fenêtres sans recouvrement
    # Une fenêtre est utilisable si tous ses instants (bornes comprises) ont une référence
    cum = np.concatenate([[0], np.cumsum(~valid)])              # nombre de NaN cumulés
    ok = (cum[starts + window + 1] - cum[starts]) == 0
    starts = starts[ok]
    Y = log_map(np.swapaxes(R[starts], -1, -2) @ R[starts + window]) * fs / window   # (M, 3)
    X = g[starts[:, None] + np.arange(window)].mean(axis=1)                          # (M, 3)
    return X, Y


def integrate_gyro(gyr, R0, fs):
    """Reconstruit l'orientation en intégrant le gyromètre, à partir d'une orientation de départ R0.

    À chaque pas, on tourne de la petite rotation mesurée pendant dt :
        R_{k+1} = R_k @ Exp(omega_k * dt)
    C'est une MULTIPLICATION sur SO(3) : le résultat reste toujours une rotation.
    Les erreurs du gyromètre s'accumulent : c'est la "dérive".

    gyr (N, 3) en rad/s, R0 (3, 3)  ->  R (N, 3, 3)
    """
    increments = exp_map(gyr[:-1] / fs)                         # (N-1, 3, 3), calculés en une fois
    R = np.empty((len(gyr), 3, 3))
    R[0] = R0
    for k in range(len(increments)):                            # la récurrence, elle, est séquentielle
        R[k + 1] = R[k] @ increments[k]
    return R
