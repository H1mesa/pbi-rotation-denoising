"""
regression.py : régressions écrites à la main en numpy (étape A).

UNE RÉGRESSION = TROIS INGRÉDIENTS
1. un MODÈLE  f(x ; theta)            -> ce qui change d'une régression à l'autre
2. une PERTE  L(theta) = moyenne (f(x ; theta) - y)^2   (erreur quadratique moyenne, MSE)
3. un OPTIMISEUR : descente de gradient  theta <- theta - lr * grad L(theta)

Chaque modèle fournit donc deux choses : `predict` (calculer f) et `gradient` (calculer grad L,
dérivé À LA MAIN). L'optimiseur `gradient_descent` est écrit une seule fois et sert à tous.

Pourquoi c'est la bonne préparation pour la suite :
- un réseau de neurones = le même schéma, avec f = composition de plusieurs fonctions
  (le gradient se calcule alors couche par couche : c'est la rétropropagation) ;
- sur SO(3), seul le PAS change : R <- R @ Exp(-lr * grad) au lieu de theta <- theta - lr * grad.

LINÉAIRE OU NON LINÉAIRE ?
Ce qui compte, c'est la dépendance en les PARAMÈTRES, pas en x :
- y = w x + b, ou même y = w2 x^2 + w1 x + b : linéaires EN theta -> solution exacte
  (moindres carrés, équations normales), la descente de gradient est facultative ;
- y = a exp(-b x) + c : b est DANS l'exponentielle -> pas de formule fermée,
  il FAUT un algorithme itératif. C'est le cas qui nous prépare aux groupes de Lie.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Outils généraux
# ---------------------------------------------------------------------------

def train_test_split(n, train_ratio=0.8, rng=None):
    """Mélange les indices 0..n-1 et les coupe en (train, test). Graine fixe = résultat reproductible."""
    rng = np.random.default_rng(rng)
    idx = rng.permutation(n)
    n_train = int(round(train_ratio * n))
    return idx[:n_train], idx[n_train:]


class Standardizer:
    """Centre-réduit une variable : z = (x - moyenne) / écart-type.

    Pourquoi ? La puissance vaut ~50 à 230 : sans mise à l'échelle, les gradients sont énormes
    pour certains paramètres et minuscules pour d'autres, et la descente oscille ou stagne.
    IMPORTANT : moyenne et écart-type sont calculés sur le TRAIN seulement, puis appliqués
    au test. Sinon le test "fuit" dans l'entraînement (fuite de données).
    """

    def fit(self, x):
        self.mean = float(np.mean(x))
        self.std = float(np.std(x))
        return self

    def transform(self, x):
        return (np.asarray(x, float) - self.mean) / self.std


def mse(y_pred, y):
    """Erreur quadratique moyenne."""
    return float(np.mean((y_pred - y) ** 2))


def r2_score(y_pred, y):
    """Coefficient de détermination : 1 = parfait, 0 = pas mieux que prédire la moyenne."""
    return 1.0 - np.sum((y_pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2)


# ---------------------------------------------------------------------------
# Modèle 1 : régression linéaire  y = w x + b,  theta = (w, b)
# ---------------------------------------------------------------------------

class LinearModel:
    name = "lineaire"
    param_names = ("w", "b")

    def predict(self, theta, x):
        w, b = theta
        return w * x + b                                        # (N,)

    def gradient(self, theta, x, y):
        """grad de L = (1/N) sum (w x + b - y)^2.

        On note r = w x + b - y (le résidu). Par la règle de dérivation d'un carré :
            dL/dw = (2/N) sum r * x
            dL/db = (2/N) sum r
        """
        r = self.predict(theta, x) - y                          # (N,)
        n = len(y)
        return np.array([2.0 / n * np.sum(r * x),
                         2.0 / n * np.sum(r)])

    def closed_form(self, x, y):
        """Solution EXACTE des moindres carrés (équations normales), pour vérifier la descente.

        On écrit y ~ X theta avec X = [x, 1] (N, 2) ; le minimum vérifie X^T X theta = X^T y.
        """
        X = np.column_stack([x, np.ones_like(x)])               # (N, 2)
        return np.linalg.solve(X.T @ X, X.T @ y)


# ---------------------------------------------------------------------------
# Modèle 2 : régression non linéaire  y = a exp(-b x) + c,  theta = (a, b, c)
# ---------------------------------------------------------------------------

class ExpDecayModel:
    """Décroissance exponentielle : la consommation baisse vite puis se stabilise vers c.

    Interprétation : c = asymptote (mpg pour une puissance très grande),
    a = amplitude au-dessus de l'asymptote (en x = 0), b = vitesse de décroissance.
    """
    name = "exponentielle"
    param_names = ("a", "b", "c")

    def predict(self, theta, x):
        a, b, c = theta
        return a * np.exp(-b * x) + c                           # (N,)

    def gradient(self, theta, x, y):
        """grad de L = (1/N) sum (a e^{-bx} + c - y)^2, avec r = résidu et e = e^{-bx}.

        Règle de la chaîne : dL/dtheta_k = (2/N) sum r * df/dtheta_k, avec
            df/da = e
            df/db = -a x e      (dérivée de e^{-bx} par rapport à b : -x e^{-bx})
            df/dc = 1
        """
        a, b, c = theta
        e = np.exp(-b * x)                                      # (N,)
        r = a * e + c - y                                       # (N,)
        n = len(y)
        return np.array([2.0 / n * np.sum(r * e),
                         2.0 / n * np.sum(r * (-a * x * e)),
                         2.0 / n * np.sum(r)])


# ---------------------------------------------------------------------------
# L'optimiseur, commun à tous les modèles
# ---------------------------------------------------------------------------

def gradient_descent(model, theta0, x, y, lr=0.1, n_iter=2000, x_test=None, y_test=None):
    """Descente de gradient à pas fixe.

    À chaque itération : calculer le gradient, faire un pas  theta <- theta - lr * grad.
    On enregistre la perte d'entraînement (et de test si fourni) pour tracer les courbes.

    Renvoie theta (n_params,) et un dict d'historique {"train": [...], "test": [...]}.
    """
    theta = np.array(theta0, dtype=float)
    history = {"train": [], "test": []}
    for _ in range(n_iter):
        history["train"].append(mse(model.predict(theta, x), y))
        if x_test is not None:
            history["test"].append(mse(model.predict(theta, x_test), y_test))
        theta = theta - lr * model.gradient(theta, x, y)
    history["train"].append(mse(model.predict(theta, x), y))
    if x_test is not None:
        history["test"].append(mse(model.predict(theta, x_test), y_test))
    return theta, history


def numerical_gradient(model, theta, x, y, h=1e-6):
    """Gradient par différences finies : (L(theta + h e_k) - L(theta - h e_k)) / 2h.

    Lent mais sans calcul à la main : sert à VÉRIFIER nos dérivées (le "gradient check").
    À refaire pour chaque nouveau modèle, et plus tard pour la rétropropagation du réseau.
    """
    theta = np.asarray(theta, float)
    g = np.zeros_like(theta)
    for k in range(len(theta)):
        d = np.zeros_like(theta)
        d[k] = h
        g[k] = (mse(model.predict(theta + d, x), y) - mse(model.predict(theta - d, x), y)) / (2 * h)
    return g
