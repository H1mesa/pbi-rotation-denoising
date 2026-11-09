"""
so3.py : les outils de base pour manipuler des rotations 3D (étape B du projet).

RAPPEL EN TROIS LIGNES
- SO(3) : l'ensemble des matrices de rotation R (3x3) telles que R^T R = I et det(R) = +1.
  C'est un espace COURBE : on y combine deux rotations en les MULTIPLIANT, jamais en les additionnant.
- so(3) : l'algèbre de Lie de SO(3) = l'espace PLAT tangent à SO(3) en l'identité.
  Ses éléments sont des matrices antisymétriques, qu'on identifie à des vecteurs w de R^3.
  Là, on a le droit d'additionner, de moyenner, de tirer du bruit gaussien.
- Exp : so(3) -> SO(3) "enroule" un vecteur sur le groupe ; Log : SO(3) -> so(3) le "déroule".

Image mentale : SO(3) est une sphère, so(3) est une feuille de papier posée dessus au pôle.
Le vecteur w = theta * u signifie "tourner d'un angle theta (radians) autour de l'axe unitaire u".

CONVENTION DE TAILLES
Toutes les fonctions acceptent des lots : un vecteur peut être (3,) ou (N, 3) ou (..., 3),
une matrice (3, 3) ou (N, 3, 3) ou (..., 3, 3). Les tailles sont notées en commentaire.
"""

import numpy as np

# Seuils numériques : en dessous de SMALL_ANGLE on utilise des développements limités,
# au-dessus de PI - NEAR_PI on utilise une formule spéciale pour Log (voir log_map).
SMALL_ANGLE = 1e-6
NEAR_PI = 1e-3


# ---------------------------------------------------------------------------
# 1. hat et vee : passer d'un vecteur de R^3 à une matrice antisymétrique, et retour
# ---------------------------------------------------------------------------

def hat(w):
    """Vecteur w = (w1, w2, w3) -> matrice antisymétrique [w]x.

    Pourquoi cette matrice ? Parce que [w]x @ v = w x v (produit vectoriel).
    C'est la façon "matricielle" d'écrire une vitesse de rotation.

        [w]x = [[  0, -w3,  w2],
                [ w3,   0, -w1],
                [-w2,  w1,   0]]

    w : (..., 3)  ->  (..., 3, 3)
    """
    w = np.asarray(w, dtype=float)
    w1, w2, w3 = w[..., 0], w[..., 1], w[..., 2]
    zero = np.zeros_like(w1)
    # On empile ligne par ligne, puis on assemble les 3 lignes
    return np.stack([
        np.stack([zero, -w3, w2], axis=-1),
        np.stack([w3, zero, -w1], axis=-1),
        np.stack([-w2, w1, zero], axis=-1),
    ], axis=-2)


def vee(W):
    """Inverse de hat : matrice antisymétrique (..., 3, 3) -> vecteur (..., 3).

    On lit simplement les trois coefficients sous la diagonale.
    """
    W = np.asarray(W, dtype=float)
    return np.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], axis=-1)


# ---------------------------------------------------------------------------
# 2. Exp : so(3) -> SO(3)  (formule de Rodrigues)
# ---------------------------------------------------------------------------

def exp_map(w):
    """Exponentielle : vecteur de rotation w -> matrice de rotation R.

    Formule de Rodrigues, avec theta = ||w|| et K = [w]x :
        R = I + A * K + B * K @ K,   A = sin(theta)/theta,   B = (1 - cos(theta))/theta^2

    Piège numérique : si theta ~ 0, on divise 0 par 0. On remplace alors A et B
    par leur développement limité (Taylor) : A ~ 1 - theta^2/6, B ~ 1/2 - theta^2/24.

    w : (..., 3)  ->  R : (..., 3, 3)
    """
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w, axis=-1)                       # (...)
    small = theta < SMALL_ANGLE
    theta_safe = np.where(small, 1.0, theta)                 # évite la division par 0

    A = np.where(small, 1.0 - theta**2 / 6.0, np.sin(theta_safe) / theta_safe)
    B = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(theta_safe)) / theta_safe**2)

    K = hat(w)                                               # (..., 3, 3)
    I = np.eye(3)
    return I + A[..., None, None] * K + B[..., None, None] * (K @ K)


# ---------------------------------------------------------------------------
# 3. Log : SO(3) -> so(3)
# ---------------------------------------------------------------------------

def log_map(R):
    """Logarithme : matrice de rotation R -> vecteur de rotation w (avec ||w|| dans [0, pi]).

    Cas général :
        theta = arccos((trace(R) - 1) / 2)
        w = theta / (2 sin(theta)) * vee(R - R^T)

    Deux cas limites à traiter à part :
    - theta ~ 0  : theta / sin(theta) -> 1 (développement : 1 + theta^2/6).
    - theta ~ pi : sin(theta) -> 0 et la partie antisymétrique R - R^T disparaît,
      donc elle ne contient plus l'axe. On lit alors l'axe u dans la partie symétrique :
      (R + R^T)/2 = cos(theta) I + (1 - cos(theta)) u u^T.

    R : (..., 3, 3)  ->  w : (..., 3)
    """
    R = np.asarray(R, dtype=float)
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)                                  # (N, 3, 3) : plus simple à indexer

    cos_t = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)  # (N,)
    theta = np.arccos(cos_t)                                 # (N,)
    v = 0.5 * vee(R - np.swapaxes(R, -1, -2))                # (N, 3) = sin(theta) * u
    w = np.zeros_like(v)                                     # (N, 3)

    small = theta < SMALL_ANGLE
    near_pi = theta > np.pi - NEAR_PI
    regular = ~small & ~near_pi

    # Cas général
    w[regular] = (theta[regular] / np.sin(theta[regular]))[:, None] * v[regular]
    # Petit angle
    w[small] = (1.0 + theta[small] ** 2 / 6.0)[:, None] * v[small]
    # Angle proche de pi
    for i in np.flatnonzero(near_pi):
        S = 0.5 * (R[i] + R[i].T)                            # partie symétrique
        uuT = (S - cos_t[i] * np.eye(3)) / (1.0 - cos_t[i])  # = u u^T
        k = np.argmax(np.diag(uuT))                          # colonne la plus fiable
        u = uuT[:, k] / np.sqrt(uuT[k, k])
        if np.dot(u, v[i]) < 0:                              # le signe vient de sin(theta) u
            u = -u
        w[i] = theta[i] * u

    return w.reshape(*batch_shape, 3)


# ---------------------------------------------------------------------------
# 4. Distance entre rotations, vérification, projection
# ---------------------------------------------------------------------------

def geodesic_distance(R1, R2):
    """Erreur géodésique d(R1, R2) = || Log(R1^T R2) ||, en radians, dans [0, pi].

    Interprétation : l'angle de la plus petite rotation qui fait passer de R1 à R2.
    C'est LA métrique du projet (et la fonction de perte des réseaux plus tard).
    Convertir en degrés pour l'affichage : np.degrees(d).

    On passe par log_map plutôt que par arccos directement, parce que arccos est
    imprécis pour les petits angles (sa dérivée explose en 1).

    R1, R2 : (..., 3, 3)  ->  (...)
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    return np.linalg.norm(log_map(np.swapaxes(R1, -1, -2) @ R2), axis=-1)


def is_rotation(R, tol=1e-8):
    """Vrai si R est dans SO(3) à tol près : R^T R = I et det(R) = 1.

    R : (..., 3, 3)  ->  booléen(s) (...)
    """
    R = np.asarray(R, dtype=float)
    orth_err = np.linalg.norm(np.swapaxes(R, -1, -2) @ R - np.eye(3), axis=(-2, -1))
    det_err = np.abs(np.linalg.det(R) - 1.0)
    return (orth_err < tol) & (det_err < tol)


def project_to_so3(M):
    """Rotation la plus proche d'une matrice 3x3 quelconque M (au sens de Frobenius).

    Méthode : SVD M = U S V^T, puis R = U diag(1, 1, det(U V^T)) V^T.
    Le terme det(...) évite de tomber sur une "réflexion" (det = -1).
    Utile pour réparer une moyenne naïve de matrices ou une sortie de réseau.

    M : (..., 3, 3)  ->  (..., 3, 3)
    """
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    d = np.sign(np.linalg.det(U @ Vt))                       # (...)
    D = np.ones(U.shape[:-1])                                # (..., 3)
    D[..., 2] = d
    return (U * D[..., None, :]) @ Vt


# ---------------------------------------------------------------------------
# 5. Génération de données : rotations aléatoires et bruit multiplicatif
# ---------------------------------------------------------------------------

def random_rotation(n, rng=None):
    """n rotations tirées uniformément sur SO(3).

    Astuce : un quaternion unitaire q = (w, x, y, z) tiré uniformément sur la sphère
    de R^4 donne une rotation uniforme. On tire 4 gaussiennes et on normalise.

    -> (n, 3, 3)
    """
    rng = np.random.default_rng(rng)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)], axis=-1),
        np.stack([2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)], axis=-1),
        np.stack([2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)], axis=-1),
    ], axis=-2)


def add_noise(R, sigma, rng=None):
    """Bruit MULTIPLICATIF : R_bruitee = R @ Exp(eps), eps ~ N(0, sigma^2 I3) dans so(3).

    C'est le modèle de bruit du projet : on tire une gaussienne dans l'espace plat so(3)
    puis on l'enroule sur SO(3) avec Exp. Le résultat est TOUJOURS une rotation,
    contrairement à R + bruit.

    R : (..., 3, 3), sigma en radians  ->  (R_bruitee (..., 3, 3), eps (..., 3))
    """
    rng = np.random.default_rng(rng)
    R = np.asarray(R, dtype=float)
    eps = rng.normal(0.0, sigma, size=R.shape[:-2] + (3,))
    return R @ exp_map(eps), eps
