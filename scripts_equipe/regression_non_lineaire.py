import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)  # AJOUT : graine fixe, les résultats sont les mêmes à chaque exécution

m = 100
X = np.random.randn(m, 1)

w = 2.0
b = 1
bruit = np.random.randn(m, 1) * 0.5

Y = w * np.exp(b*X) + bruit

#%%
# AJOUT : découpage 80 % entraînement / 20 % test.
# On mélange les indices, on met 20 points de côté (jamais vus pendant l'apprentissage),
# et X, Y ne contiennent plus que les 80 points d'entraînement.
indices = np.random.permutation(m)
n_train = int(0.8 * m)
X_test, Y_test = X[indices[n_train:]], Y[indices[n_train:]]
X, Y = X[indices[:n_train]], Y[indices[:n_train]]

#%%

W = 0.7
B = 0.3

def predict(X, W, B):
    return W*np.exp(B*X)

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

    dW = np.sum(np.exp(B*X)*error)/m
    dB = np.sum(W*X*np.exp(B*X)*error)/m

    return dW, dB

#%%
# AJOUT : vérification du gradient par différences finies.
# La dérivée se rapproche de (loss(W + h) - loss(W - h)) / 2h quand h est petit.
# C'est la vérification la plus utile ici : dB contient la règle de la chaîne
# (dérivée de exp(B x) par rapport à B = x exp(B x)), facile à rater.
def verif_gradient(X, Y, W, B, h=1e-6):
    dW, dB = grad(X, Y, W, B)
    dW_num = (loss(X, Y, W + h, B) - loss(X, Y, W - h, B)) / (2 * h)
    dB_num = (loss(X, Y, W, B + h) - loss(X, Y, W, B - h)) / (2 * h)
    print(f"Vérification du gradient : dW {dW:.6f} vs {dW_num:.6f} | dB {dB:.6f} vs {dB_num:.6f}")
    print("Gradient correct :", np.isclose(dW, dW_num, rtol=1e-5) and np.isclose(dB, dB_num, rtol=1e-5))

verif_gradient(X, Y, W, B)

#%%
learning_rate=0.01  # MODIFIÉ : 0.0001 -> 0.01. Avec 0.0001, après 20000 itérations on avait seulement
                    # W = 1.92, B = 1.03 (coût 0.138) : pas fini de converger. Avec 0.01 : W = 2.05, B = 0.99 (coût 0.130).
nb_iter = 20000

for i in range(nb_iter):
    dW, dB = grad(X, Y, W, B)

    W -= learning_rate * dW
    B -= learning_rate * dB

    if i % 1000 == 0:   # MODIFIÉ : affichage toutes les 1000 itérations au lieu de 100
        current_loss = loss(X, Y, W, B)
        print(f"Itération {i} | Coût : {current_loss:.4f} | W : {W:.4f} | B : {B:.4f}")

print(f"\nRésultat final après {nb_iter} itérations :")
print(f"W estimé : {W:.4f} (Attendu : {w})")
print(f"B estimé  : {B:.4f} (Attendu : {b})")

#%%
# AJOUT : erreur sur le jeu de test (20 points jamais vus).
# Si elle est proche de l'erreur d'entraînement, le modèle généralise bien.
print(f"\nCoût entraînement : {loss(X, Y, W, B):.4f} | Coût test : {loss(X_test, Y_test, W, B):.4f}")

# AJOUT : pas de solution exacte ici. B est DANS l'exponentielle : le modèle n'est pas linéaire
# en ses paramètres, aucune formule ne donne directement le minimum, il FAUT itérer.
# On compare donc aux vrais paramètres, et au coût obtenu avec eux : la descente ne peut
# pas faire beaucoup mieux que ce coût (il reste le bruit).
print(f"Coût d'entraînement avec les vrais paramètres (w={w}, b={b}) : {loss(X, Y, w, b):.4f}")

#%%
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, color='blue', alpha=0.6)
plt.scatter(X_test, Y_test, color='orange', marker='s', label='test (20 %)')  # AJOUT : points de test

X_sorted = np.sort(X, axis=0)
Y_pred_sorted = predict(X_sorted, W, B)
plt.plot(X_sorted, Y_pred_sorted, color='red', linewidth=2)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Régression Exponentielle Pure')
plt.legend()  # AJOUT : légende pour distinguer les points de test
plt.grid(True)
plt.show()
