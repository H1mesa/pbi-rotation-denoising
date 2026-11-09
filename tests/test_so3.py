"""
Tests de src/so3.py. Lancer depuis le dossier du projet :  python -m pytest -v

Chaque test vérifie une PROPRIÉTÉ mathématique (pas juste "ça ne plante pas").
SciPy sert de référence indépendante pour Exp et Log.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from src.so3 import (hat, vee, exp_map, log_map, geodesic_distance, is_rotation,
                     project_to_so3, random_rotation, add_noise)

SEED = 0  # graine fixe : les tests donnent toujours le même résultat


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


# --- hat / vee -------------------------------------------------------------

def test_hat_est_antisymetrique_et_vee_inverse(rng):
    w = rng.normal(size=(50, 3))                                # (50, 3)
    W = hat(w)                                                  # (50, 3, 3)
    assert np.allclose(W, -np.swapaxes(W, -1, -2))              # W^T = -W
    assert np.allclose(vee(W), w)                               # vee(hat(w)) = w


def test_hat_donne_le_produit_vectoriel(rng):
    w, v = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(hat(w) @ v, np.cross(w, v))


# --- Exp -------------------------------------------------------------------

def test_exp_donne_une_rotation(rng):
    w = rng.normal(scale=2.0, size=(200, 3))
    assert is_rotation(exp_map(w)).all()


def test_exp_identique_a_scipy(rng):
    w = rng.normal(scale=2.0, size=(200, 3))
    assert np.allclose(exp_map(w), Rot.from_rotvec(w).as_matrix())


def test_exp_de_zero_et_petits_angles():
    assert np.allclose(exp_map(np.zeros(3)), np.eye(3))
    w = np.array([1e-9, -2e-9, 3e-10])                          # zone du développement limité
    assert np.allclose(exp_map(w), Rot.from_rotvec(w).as_matrix(), atol=1e-14)


# --- Log -------------------------------------------------------------------

def test_log_exp_est_identite_pour_angle_inferieur_a_pi(rng):
    # On tire des axes et des angles dans [0, pi[ : Log(Exp(w)) doit redonner w
    u = rng.normal(size=(500, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    theta = rng.uniform(0, np.pi - 1e-2, size=500)
    w = theta[:, None] * u
    assert np.allclose(log_map(exp_map(w)), w, atol=1e-9)


def test_exp_log_est_identite_sur_rotations_aleatoires(rng):
    R = random_rotation(500, rng)
    assert np.allclose(exp_map(log_map(R)), R, atol=1e-9)


def test_log_identique_a_scipy(rng):
    R = random_rotation(200, rng)
    assert np.allclose(log_map(R), Rot.from_matrix(R).as_rotvec(), atol=1e-9)


@pytest.mark.parametrize("theta", [0.0, 1e-8, 1e-4, np.pi - 1e-2, np.pi - 1e-5, np.pi])
def test_log_cas_limites(theta):
    u = np.array([1.0, 2.0, -2.0]) / 3.0                        # axe unitaire
    R = exp_map(theta * u)
    w = log_map(R)
    # On compare les ROTATIONS : à theta = pi, w et -w donnent la même rotation
    assert np.allclose(exp_map(w), R, atol=1e-7)
    assert np.isclose(np.linalg.norm(w), theta, atol=1e-7)


def test_log_accepte_un_lot_multidimensionnel(rng):
    R = random_rotation(12, rng).reshape(3, 4, 3, 3)
    assert log_map(R).shape == (3, 4, 3)


# --- Distance géodésique ----------------------------------------------------

def test_distance_vaut_l_angle_de_la_perturbation(rng):
    R = random_rotation(100, rng)
    u = rng.normal(size=(100, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    theta = rng.uniform(0, np.pi - 1e-2, size=100)
    R2 = R @ exp_map(theta[:, None] * u)
    assert np.allclose(geodesic_distance(R, R2), theta, atol=1e-9)


def test_distance_symetrique_et_invariante(rng):
    R1, R2, Q = random_rotation(3 * 50, rng).reshape(3, 50, 3, 3)
    d = geodesic_distance(R1, R2)
    assert np.allclose(d, geodesic_distance(R2, R1))
    # Invariance : tourner les deux rotations par la même Q ne change pas leur écart
    assert np.allclose(d, geodesic_distance(Q @ R1, Q @ R2))
    assert np.allclose(geodesic_distance(R1, R1), 0.0)


# --- Projection, génération, bruit ------------------------------------------

def test_projection_repare_une_matrice_perturbee(rng):
    R = random_rotation(50, rng)
    M = R + 0.01 * rng.normal(size=R.shape)                     # n'est plus une rotation
    assert not is_rotation(M).any()
    P = project_to_so3(M)
    assert is_rotation(P).all()
    assert (geodesic_distance(P, R) < 0.05).all()              # et reste proche de R


def test_projection_d_une_rotation_ne_change_rien(rng):
    R = random_rotation(20, rng)
    assert np.allclose(project_to_so3(R), R)


def test_random_rotation_est_dans_so3(rng):
    assert is_rotation(random_rotation(1000, rng)).all()


def test_bruit_additif_casse_so3_mais_pas_le_multiplicatif(rng):
    """LE test qui résume pourquoi le projet existe."""
    R = random_rotation(100, rng)
    R_additif = R + rng.normal(0, 0.1, size=R.shape)            # FAUX physiquement
    R_mult, _ = add_noise(R, sigma=0.1, rng=rng)                # correct
    assert not is_rotation(R_additif).any()
    assert is_rotation(R_mult).all()


def test_statistique_du_bruit(rng):
    # eps ~ N(0, sigma^2 I3)  =>  E[||eps||^2] = 3 sigma^2, et d(R, R_bruitee) = ||eps|| si ||eps|| < pi
    sigma = 0.1
    R = random_rotation(20000, rng)
    R_b, eps = add_noise(R, sigma, rng)
    d = geodesic_distance(R, R_b)
    assert np.allclose(d, np.linalg.norm(eps, axis=1), atol=1e-9)
    assert np.isclose(np.mean(d**2), 3 * sigma**2, rtol=0.03)
