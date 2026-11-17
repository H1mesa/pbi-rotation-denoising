"""
regression_arthur.py : régression linéaire écrite à la main (code d'Arthur Ritz, complété).

Le cœur est celui d'Arthur : predict, loss, grad et la boucle de descente de gradient.
Ajouts par rapport à sa première version :
- graine aléatoire fixe (résultats reproductibles) ;
- découpage 80 % entraînement / 20 % test (on mesure l'erreur sur des données jamais vues) ;
- comparaison avec la solution EXACTE des moindres carrés (preuve que la descente converge au bon endroit) ;
- plusieurs sorties possibles : Y peut être (m, 1) ou (m, p). W est alors (n, p) et B (1, p).
  Les formules ne changent pas, c'est l'intérêt d'avoir tout écrit avec des matrices.
- une fonction `fit` réutilisable sur d'autres données (make_friedman1, BROAD).

Convention de la perte (choix d'Arthur) : L = somme des carrés / (2m).
Le 1/2 simplifie la dérivée : dL/dW = X^T (XW + B - Y) / m, sans facteur 2.
"""

import numpy as np


def predict(X, W, B):
    """Prédiction linéaire : X (m, n) @ W (n, p) + B (1, p) -> (m, p)."""
    return X @ W + B


def loss(X, Y, W, B):
    """Erreur quadratique / (2m), sommée sur toutes les sorties."""
    m = X.shape[0]
    predictions = predict(X, W, B)
    return np.sum((predictions - Y) ** 2) / (2 * m)


def grad(X, Y, W, B):
    """Gradient de la perte par rapport à W et B.

    error = XW + B - Y   (m, p)
    dW = X^T error / m   (n, p)
    dB = somme des erreurs / m, colonne par colonne   (1, p)
    """
    m = X.shape[0]
    error = predict(X, W, B) - Y
    dW = (X.T @ error) / m
    dB = np.sum(error, axis=0, keepdims=True) / m
    return dW, dB


def fit(X, Y, learning_rate=0.1, nb_iter=1000, X_test=None, Y_test=None, verbose=False):
    """Boucle de descente de gradient d'Arthur. Renvoie W, B et l'historique des pertes."""
    Y = Y.reshape(len(Y), -1)                                   # (m,) -> (m, 1) si besoin
    W = np.zeros((X.shape[1], Y.shape[1]))
    B = np.zeros((1, Y.shape[1]))
    historique = {"train": [], "test": []}
    for i in range(nb_iter):
        dW, dB = grad(X, Y, W, B)
        W -= learning_rate * dW
        B -= learning_rate * dB
        historique["train"].append(loss(X, Y, W, B))
        if X_test is not None:
            historique["test"].append(loss(X_test, Y_test.reshape(len(Y_test), -1), W, B))
        if verbose and i % 100 == 0:
            print(f"Itération {i} | Coût : {historique['train'][-1]:.4f}")
    return W, B, historique


def solution_exacte(X, Y):
    """Moindres carrés exacts (équations normales) : sert à VÉRIFIER la descente de gradient."""
    Y = Y.reshape(len(Y), -1)
    Xb = np.column_stack([X, np.ones(len(X))])                  # colonne de 1 pour B
    theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ Y)                # (n + 1, p)
    return theta[:-1], theta[-1:]                               # W (n, p), B (1, p)


def split_80_20(m, seed=42):
    """Mélange les indices et les coupe en 80 % / 20 %."""
    idx = np.random.default_rng(seed).permutation(m)
    n_train = int(round(0.8 * m))
    return idx[:n_train], idx[n_train:]


if __name__ == "__main__":
    # Exemple d'origine d'Arthur : Y = 3X + 4 + bruit, désormais reproductible et évalué sur un test.
    rng = np.random.default_rng(0)                              # graine fixe
    m = 100
    X = rng.normal(size=(m, 1))
    w, b = 3, 4
    Y = w * X + b + rng.normal(size=(m, 1))

    tr, te = split_80_20(m)
    W, B, hist = fit(X[tr], Y[tr], learning_rate=0.1, nb_iter=200, X_test=X[te], Y_test=Y[te], verbose=True)
    W_ex, B_ex = solution_exacte(X[tr], Y[tr])

    print(f"\nW estimé : {W[0, 0]:.4f} | exact : {W_ex[0, 0]:.4f} | attendu : {w}")
    print(f"B estimé : {B[0, 0]:.4f} | exact : {B_ex[0, 0]:.4f} | attendu : {b}")
    print(f"Perte test : {hist['test'][-1]:.4f}  (bruit de variance 1 -> plancher ~ 0.5 avec le 1/2m)")
