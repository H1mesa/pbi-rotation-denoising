"""Tests de src/regression_arthur.py (code d'Arthur complété)."""

import numpy as np

from src.regression_arthur import predict, loss, grad, fit, solution_exacte, split_80_20


def test_gradient_d_arthur_egal_differences_finies():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(50, 3)), rng.normal(size=(50, 2))
    W, B = rng.normal(size=(3, 2)), rng.normal(size=(1, 2))
    dW, dB = grad(X, Y, W, B)
    h = 1e-6
    for i in range(3):
        for j in range(2):
            E = np.zeros_like(W)
            E[i, j] = h
            num = (loss(X, Y, W + E, B) - loss(X, Y, W - E, B)) / (2 * h)
            assert np.isclose(dW[i, j], num, rtol=1e-5)
    for j in range(2):
        E = np.zeros_like(B)
        E[0, j] = h
        assert np.isclose(dB[0, j], (loss(X, Y, W, B + E) - loss(X, Y, W, B - E)) / (2 * h), rtol=1e-5)


def test_descente_retrouve_la_solution_exacte_avec_plusieurs_sorties():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    W_vrai = np.array([[1.0, 0.0], [2.0, -1.0], [0.0, 3.0]])
    Y = X @ W_vrai + np.array([[0.5, -2.0]]) + 0.1 * rng.normal(size=(300, 2))
    W, B, hist = fit(X, Y, learning_rate=0.1, nb_iter=2000)
    W_ex, B_ex = solution_exacte(X, Y)
    assert np.allclose(W, W_ex, atol=1e-8) and np.allclose(B, B_ex, atol=1e-8)
    assert hist["train"][-1] < hist["train"][0]


def test_sortie_unique_en_vecteur():
    X = np.linspace(0, 1, 20)[:, None]
    W, B, _ = fit(X, 2 * X[:, 0] + 1, learning_rate=0.5, nb_iter=5000)
    assert np.allclose(predict(X, W, B)[:, 0], 2 * X[:, 0] + 1, atol=1e-6)


def test_split_80_20():
    tr, te = split_80_20(100)
    assert len(tr) == 80 and len(te) == 20 and set(tr).isdisjoint(te)


# --- Données BROAD -----------------------------------------------------------

import pytest
from src.broad import DEFAULT_DIR, list_trials, load_trial, angular_rate_pairs, integrate_gyro
from src.so3 import exp_map, is_rotation, geodesic_distance


def test_integration_d_une_vitesse_constante():
    # Tourner à vitesse constante w pendant t secondes = Exp(w t)
    w, fs, n = np.array([0.3, -0.2, 0.5]), 100.0, 201
    R = integrate_gyro(np.tile(w, (n, 1)), np.eye(3), fs)
    assert is_rotation(R, tol=1e-9).all()
    assert np.allclose(R[-1], exp_map(w * (n - 1) / fs), atol=1e-10)


@pytest.mark.skipif(not list_trials(), reason="data/broad absent")
def test_chargement_broad_et_paires():
    t = load_trial(list_trials()[0])
    assert t["gyr"].shape[1] == 3 and np.isclose(t["fs"], 285.714, atol=0.01)
    assert is_rotation(t["R_ref"][t["valid"]], tol=1e-6).all()
    X, Y = angular_rate_pairs(t)
    assert X.shape == Y.shape and not np.isnan(Y).any()
    # Le gyromètre et la vitesse déduite des caméras doivent être très corrélés
    assert np.corrcoef(X[:, 2], Y[:, 2])[0, 1] > 0.95
