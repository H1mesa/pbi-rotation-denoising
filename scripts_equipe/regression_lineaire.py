import numpy as np

np.random.seed(0)  # AJOUT : graine fixe, les résultats sont les mêmes à chaque exécution

m = 100
X=np.random.randn(m,1)

# Y = wX + b
w = 3
b = 4
bruit = np.random.randn(m,1)

Y = w * X + b + bruit

#%%
# AJOUT : découpage 80 % entraînement / 20 % test.
# On mélange les indices, on met 20 points de côté (jamais vus pendant l'apprentissage),
# et X, Y ne contiennent plus que les 80 points d'entraînement.
indices = np.random.permutation(m)
n_train = int(0.8 * m)
X_test, Y_test = X[indices[n_train:]], Y[indices[n_train:]]
X, Y = X[indices[:n_train]], Y[indices[:n_train]]

#%%
W = np.zeros((X.shape[1], 1))
B = 0

def predict(X, W, B):
    return X @ W + B

#%%
def loss(X, Y, W, B):
    m = X.shape[0]
    predictions = predict(X, W, B)
    cost = (np.sum((predictions - Y)**2))/(2*m)
    return cost

#%%
def grad(X, Y, W, B):
    m = X.shape[0]
    predictions = predict(X, W, B)
    error = predictions - Y

    dW = (X.T @ error)/m
    dB = np.sum(error)/m

    return dW, dB

#%%
# AJOUT : vérification du gradient par différences finies.
# La dérivée se rapproche de (loss(W + h) - loss(W - h)) / 2h quand h est petit.
# Si ces valeurs sont égales à celles de grad(), nos dérivées écrites à la main sont justes.
def verif_gradient(X, Y, W, B, h=1e-6):
    dW, dB = grad(X, Y, W, B)
    dW_num = np.zeros_like(W)
    for i in range(W.shape[0]):
        E = np.zeros_like(W)
        E[i, 0] = h
        dW_num[i, 0] = (loss(X, Y, W + E, B) - loss(X, Y, W - E, B)) / (2 * h)
    dB_num = (loss(X, Y, W, B + h) - loss(X, Y, W, B - h)) / (2 * h)
    print(f"Vérification du gradient : dW {dW.ravel()} vs {dW_num.ravel()} | dB {dB:.6f} vs {dB_num:.6f}")
    print("Gradient correct :", np.allclose(dW, dW_num, rtol=1e-5) and np.isclose(dB, dB_num, rtol=1e-5))

verif_gradient(X, Y, W, B)

#%%
learning_rate=0.1   # MODIFIÉ : 0.001 -> 0.1, la descente converge 100 fois plus vite et reste stable
nb_iter = 1000      # MODIFIÉ : 10000 -> 1000, suffisant avec ce pas

for i in range(nb_iter):
    dW, dB = grad(X, Y, W, B)

    W -= learning_rate * dW
    B -= learning_rate * dB

    if i % 100 == 0:
        current_loss = loss(X, Y, W, B)
        print(f"Itération {i} | Coût : {current_loss:.4f} | W : {W[0,0]:.4f} | B : {B:.4f}")

print(f"\nRésultat final après {nb_iter} itérations :")
print(f"W estimé : {W[0,0]:.4f} (Attendu : {w})")
print(f"B estimé : {B:.4f} (Attendu : {b})")

#%%
# AJOUT : erreur sur le jeu de test (20 points jamais vus).
# Si elle est proche de l'erreur d'entraînement, le modèle généralise bien.
print(f"\nCoût entraînement : {loss(X, Y, W, B):.4f} | Coût test : {loss(X_test, Y_test, W, B):.4f}")

# AJOUT : comparaison avec la solution EXACTE des moindres carrés.
# Le modèle est linéaire en W et B : le minimum se calcule directement (équations normales).
# Si la descente de gradient a convergé, elle doit donner les mêmes valeurs.
X1 = np.hstack([X, np.ones((X.shape[0], 1))])        # colonne de 1 pour B
theta_exact = np.linalg.solve(X1.T @ X1, X1.T @ Y)
print(f"Solution exacte : W = {theta_exact[0,0]:.4f} | B = {theta_exact[1,0]:.4f}")
