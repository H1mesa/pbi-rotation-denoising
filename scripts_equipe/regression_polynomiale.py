import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)  # AJOUT : graine fixe, les résultats sont les mêmes à chaque exécution

m = 100
X = np.random.randn(m, 1)

w1 = 2.0
w2 = -3.0
b = 5.0
bruit = np.random.randn(m, 1) * 0.5

Y = w1 * (X ** 2) + w2 * X + b + bruit

#%%
# AJOUT : découpage 80 % entraînement / 20 % test.
# On mélange les indices, on met 20 points de côté (jamais vus pendant l'apprentissage),
# et X, Y ne contiennent plus que les 80 points d'entraînement.
indices = np.random.permutation(m)
n_train = int(0.8 * m)
X_test, Y_test = X[indices[n_train:]], Y[indices[n_train:]]
X, Y = X[indices[:n_train]], Y[indices[:n_train]]
Xp_test = np.hstack([X_test**2, X_test])  # AJOUT : mêmes variables (x², x) pour le test

#%%
Xp = np.hstack([X**2, X])

W = np.zeros((Xp.shape[1], 1))
B = 0

def predict(Xp, W, B):
    return Xp @ W + B

#%%
def loss(Xp, Y, W, B):
    m = Xp.shape[0]
    predictions = predict(Xp, W, B)
    cost = (np.sum((predictions - Y)**2))/(2*m)
    return cost

#%%
def grad(Xp, Y, W, B):
    m = Xp.shape[0]
    predictions = predict(Xp, W, B)
    error = predictions - Y

    dW = (Xp.T @ error)/m
    dB = np.sum(error)/m

    return dW, dB

#%%
# AJOUT : vérification du gradient par différences finies.
# La dérivée se rapproche de (loss(W + h) - loss(W - h)) / 2h quand h est petit.
# Si ces valeurs sont égales à celles de grad(), nos dérivées écrites à la main sont justes.
def verif_gradient(Xp, Y, W, B, h=1e-6):
    dW, dB = grad(Xp, Y, W, B)
    dW_num = np.zeros_like(W)
    for i in range(W.shape[0]):
        E = np.zeros_like(W)
        E[i, 0] = h
        dW_num[i, 0] = (loss(Xp, Y, W + E, B) - loss(Xp, Y, W - E, B)) / (2 * h)
    dB_num = (loss(Xp, Y, W, B + h) - loss(Xp, Y, W, B - h)) / (2 * h)
    print(f"Vérification du gradient : dW {dW.ravel()} vs {dW_num.ravel()} | dB {dB:.6f} vs {dB_num:.6f}")
    print("Gradient correct :", np.allclose(dW, dW_num, rtol=1e-5) and np.isclose(dB, dB_num, rtol=1e-5))

verif_gradient(Xp, Y, W, B)

#%%
learning_rate=0.1   # MODIFIÉ : 0.001 -> 0.1, convergence beaucoup plus rapide et toujours stable
nb_iter = 2000      # MODIFIÉ : 10000 -> 2000, suffisant avec ce pas

for i in range(nb_iter):
    dW, dB = grad(Xp, Y, W, B)

    W -= learning_rate * dW
    B -= learning_rate * dB

    if i % 100 == 0:
        current_loss = loss(Xp, Y, W, B)
        print(f"Itération {i} | Coût : {current_loss:.4f} | W : {W[0,0]:.4f} | B : {B:.4f}")

print(f"\nRésultat final après {nb_iter} itérations :")
print(f"W1 estimé : {W[0,0]:.4f} (Attendu : {w1})")
print(f"W2 estimé : {W[1,0]:.4f} (Attendu : {w2})")
print(f"B estimé  : {B:.4f} (Attendu : {b})")

#%%
# AJOUT : erreur sur le jeu de test (20 points jamais vus).
# Si elle est proche de l'erreur d'entraînement, le modèle généralise bien.
print(f"\nCoût entraînement : {loss(Xp, Y, W, B):.4f} | Coût test : {loss(Xp_test, Y_test, W, B):.4f}")

# AJOUT : comparaison avec la solution EXACTE des moindres carrés.
# y = w1 x² + w2 x + b est courbe en x, mais LINÉAIRE en (w1, w2, b) :
# la formule exacte existe encore. La descente de gradient doit donner les mêmes valeurs.
Xp1 = np.hstack([Xp, np.ones((Xp.shape[0], 1))])     # colonne de 1 pour B
theta_exact = np.linalg.solve(Xp1.T @ Xp1, Xp1.T @ Y)
print(f"Solution exacte : W1 = {theta_exact[0,0]:.4f} | W2 = {theta_exact[1,0]:.4f} | B = {theta_exact[2,0]:.4f}")

#%%
plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color='blue', alpha=0.6)
plt.scatter(X_test, Y_test, color='orange', marker='s', label='test (20 %)')  # AJOUT : points de test

X_sorted_idx = np.argsort(X[:, 0])
X_sorted = X[X_sorted_idx]
Xp_sorted = np.hstack([X_sorted**2, X_sorted])
Y_pred_sorted = predict(Xp_sorted, W, B)
plt.plot(X_sorted, Y_pred_sorted, color='red', linewidth=2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Régression Polynomiale')
plt.legend()  # AJOUT : légende pour distinguer les points de test
plt.grid(True)
plt.show()
