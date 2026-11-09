"""
Tests de src/regression.py et src/auto_mpg.py. Lancer : python -m pytest -v
"""

import numpy as np
import pytest

from src.regression import (LinearModel, ExpDecayModel, Standardizer, gradient_descent,
                            numerical_gradient, train_test_split, mse, r2_score)
from src.auto_mpg import DEFAULT_PATH, load_raw, clean

SEED = 0


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


# --- Gradient check : nos dérivées à la main = différences finies ----------

@pytest.mark.parametrize("model", [LinearModel(), ExpDecayModel()], ids=lambda m: m.name)
def test_gradient_a_la_main_egal_differences_finies(model, rng):
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    for _ in range(5):
        theta = rng.normal(size=len(model.param_names))
        assert np.allclose(model.gradient(theta, x, y),
                           numerical_gradient(model, theta, x, y), rtol=1e-5, atol=1e-7)


# --- Convergence ------------------------------------------------------------

def test_descente_lineaire_retrouve_la_solution_exacte(rng):
    x = rng.normal(size=200)
    y = 3.0 * x - 1.0 + 0.1 * rng.normal(size=200)
    model = LinearModel()
    theta, history = gradient_descent(model, [0.0, 0.0], x, y, lr=0.1, n_iter=500)
    assert np.allclose(theta, model.closed_form(x, y), atol=1e-8)
    assert history["train"][-1] <= history["train"][0]


def test_descente_exponentielle_retrouve_des_parametres_connus(rng):
    # Données synthétiques sans bruit : on connaît la vérité, on doit la retrouver
    x = rng.uniform(-1.5, 2.5, size=300)
    vrai = np.array([1.5, 0.6, -1.8])
    model = ExpDecayModel()
    y = model.predict(vrai, x)
    theta, _ = gradient_descent(model, [1.0, 0.5, 0.0], x, y, lr=0.1, n_iter=20000)
    assert np.allclose(theta, vrai, atol=1e-3)


def test_initialisation_a_zero_bloque_le_parametre_b(rng):
    # Avec a = 0, df/db = -a x e^{-bx} = 0 : le gradient en b est nul au départ
    x, y = rng.normal(size=50), rng.normal(size=50)
    assert ExpDecayModel().gradient(np.zeros(3), x, y)[1] == 0.0


# --- Outils -----------------------------------------------------------------

def test_split_80_20_disjoint_et_complet():
    tr, te = train_test_split(100, 0.8, rng=42)
    assert len(tr) == 80 and len(te) == 20
    assert set(tr).isdisjoint(te) and set(tr) | set(te) == set(range(100))
    tr2, _ = train_test_split(100, 0.8, rng=42)
    assert np.array_equal(tr, tr2)                              # même graine, même découpage


def test_standardizer_utilise_les_statistiques_du_train(rng):
    x_train, x_test = rng.normal(5, 2, size=500), rng.normal(5, 2, size=100)
    s = Standardizer().fit(x_train)
    z = s.transform(x_train)
    assert np.isclose(z.mean(), 0) and np.isclose(z.std(), 1)
    assert np.allclose(s.transform(x_test), (x_test - x_train.mean()) / x_train.std())


def test_metriques():
    y = np.array([1.0, 2.0, 3.0])
    assert mse(y, y) == 0 and r2_score(y, y) == 1
    assert np.isclose(r2_score(np.full(3, 2.0), y), 0)          # prédire la moyenne -> R² = 0


# --- Données réelles --------------------------------------------------------

@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="data/auto-mpg.data absent")
def test_nettoyage_auto_mpg():
    df, rapport = clean(load_raw())
    assert rapport["lignes_brutes"] == 398
    assert rapport["apres_plausibilite"] == len(df) == 392       # 6 puissances manquantes retirées
    assert not df[["mpg", "horsepower"]].isna().any().any()
    assert not df.duplicated().any()
