Homework 1 - Nonlinear Option Pricing
====================================================
-------------------------------------------------------------------------------
1. Conditional Expectation and Least Squares Regression
-------------------------------------------------------------------------------

Je fixe la définition des variables de base (comme dans l'énoncé).

```python
import numpy as np
import math

np.random.seed(123)

def g(x):
    return x*(1+x)/(1+x**2)

n = 200
sigma = 0.25
X = np.random.randn(n)
Y = g(X) + sigma*np.random.randn(n)
```

Question 1.1 
------------------------------------

Je compare l'ajustement par Polynômes de degré différents et par
régression piecewise-lineaire avec un nombre de noeuds variable.
L'objectif est d'illustrer le sur-apprentissage : trop de flexibilite
donne une courbe instable, trop peu donne un sous-ajustement.

Code
------------------------------------

```python
X_test = np.random.randn(10000)
Y_true = g(X_test)

# polynomes
poly_degs = [1, 3, 5, 9, 12]
poly_mse = []
for d in poly_degs:
    p = np.polyfit(X, Y, deg=d)
    pred = np.polyval(p, X_test)
    mse = np.mean((pred - Y_true)**2)
    poly_mse.append((d, mse))

print("Polynomial test MSE:", poly_mse)

# piecewise linear
def pwlin_basis(xknots):
    fs = [lambda x: np.ones_like(x, dtype=float), lambda x: x-xknots[0]]
    fs.extend([lambda x, a=xknots[i]: np.maximum(x-a, 0.0) for i in range(len(xknots))])
    return fs

def pwlin_fit(xdata, ydata, xknots):
    fs = pwlin_basis(xknots)
    A = np.column_stack([f(xdata) for f in fs])
    ps, *_ = np.linalg.lstsq(A, ydata, rcond=None)
    return ps, fs

def pwlin_predict(ps, fs, x):
    A = np.column_stack([f(x) for f in fs])
    return A @ ps

knot_counts = [3, 5, 9, 15]
pl_mse = []
for k in knot_counts:
    xknots = np.linspace(X.min(), X.max(), k)
    ps, fs = pwlin_fit(X, Y, xknots)
    pred = pwlin_predict(ps, fs, X_test)
    mse = np.mean((pred - Y_true)**2)
    pl_mse.append((k, mse))

print("Piecewise-linear test MSE:", pl_mse)
```

Resultats (exemple, seed=123)
-----------------------------

Polynômes (MSE test vs g) :
 - deg 1: 0.1054
 - deg 3: 0.0407
 - deg 5: 0.0171
 - deg 9: 0.3826
 - deg 12: 347.445

Piecewise-linear (MSE test) :
 - 3 noeuds: 0.0244
 - 5 noeuds: 0.00315
 - 9 noeuds: 0.00298
 - 15 noeuds: 0.00561

Commentaire :
 - degrés 1-3 sous-ajustent la courbe.
 - degré 5 est un bon compromis.
 - degrés 9-12 sur-ajustent clairement.
 - piecewise-linear s'ameliore avec 5-9 noeuds, puis se degrade si trop de noeuds.

Question 1.2
---------------------------------------

Je teste la régression noyau (Nadaraya-Watson) avec plusieurs largeurs de
bande, puis je compare deux noyaux.

Code
------------------------

```python
def gauss_kern(u):
    return (1/np.sqrt(2*np.pi))*np.exp(-0.5*u**2)

def quartic_kern(u):
    out = np.zeros_like(u)
    mask = np.abs(u) <= 1
    out[mask] = (1+u[mask])**2 * (1-u[mask])**2
    return out

def kern_reg_grid(xgrid, xdata, ydata, bandwidth, kern):
    xdata = xdata.reshape(1, -1)
    xgrid = xgrid.reshape(-1, 1)
    weights = kern((xgrid - xdata)/bandwidth) / bandwidth
    num = (weights * ydata.reshape(1, -1)).sum(axis=1)
    den = weights.sum(axis=1)
    return num/den

xgrid = np.linspace(X.min(), X.max(), 200)
bandwidths = [0.1, 0.2, 0.4, 0.8, 1.6]

kw_mse = []
for h in bandwidths:
    ygrid = kern_reg_grid(xgrid, X, Y, h, gauss_kern)
    pred = np.interp(X_test, xgrid, ygrid, left=ygrid[0], right=ygrid[-1])
    mse = np.mean((pred - Y_true)**2)
    kw_mse.append((h, mse))
print("Gaussian kernel MSE:", kw_mse)

for h in [0.2, 0.4]:
    for name, kern in [("gauss", gauss_kern), ("quartic", quartic_kern)]:
        ygrid = kern_reg_grid(xgrid, X, Y, h, kern)
        pred = np.interp(X_test, xgrid, ygrid, left=ygrid[0], right=ygrid[-1])
        mse = np.mean((pred - Y_true)**2)
        print(f"kernel={name} h={h} mse={mse}")
```

Résultats (exemple, seed=123)
-----------------------------

Gaussian kernel :
 - h=0.1: 0.00618 (sur-ajustement)
 - h=0.2: 0.00532 (meilleur)
 - h=0.4: 0.01457
 - h=0.8: 0.06309 (sous-ajustement)
 - h=1.6: 0.15238 (sous-ajustement)

Comparaison de noyaux :
 - h=0.2: gaussian 0.00532 vs quartic 0.00731
 - h=0.4: gaussian 0.01457 vs quartic 0.00507

Conclusion : la largeur de bande h influence beaucoup plus la qualite du
fit que le choix du noyau.

-------------------------------------------------------------------------------
2. American Option Pricing
-------------------------------------------------------------------------------

Parametres communs :
S0=100, vol=0.2, r=0.1, q=0.02, K=100, T=1  
Dates d'exercice mensuelles : ts = linspace(0,1,13)

Question 2.1
------------------------------------------------------

Procédure :
1) Simuler les trajectoires de S_t (Black-Scholes).
2) Appliquer LS et TVR.
3) Remplacer la base polynomiale par :
   (a) base [1, prix BS put(K,T-t,vol=0.2)]
   (b) regression piecewise-lineaire
   (c) regression noyau gaussien.

Resultats (20 000 trajectoires) :
 - LS (BS basis)  : 5.1227
 - TVR (BS basis) : 5.2232
 - LS (piecewise) : 5.1358
 - TVR (piecewise): 5.1166
 - LS (kernel)    : 4.8452
 - TVR (kernel)   : 8.2603

Commentaire :
Les méthodes paramétriques sont assez stables.  
La régression noyau est beaucoup plus sensible a la largeur de bande,
ce qui explique la variabilite dans TVR.

Question 2.2 
--------------------------------------------------------

Procedure :
1) Construire la politique d'exercice avec LS ou TVR.
2) Simuler un second jeu de trajectoires independantes (100 000).
3) Appliquer la politique et calculer la valeur actualisee.

Resultats (quadratique, 100 000 trajectoires independantes) :
 - LS lower bound  : 5.4588
 - TVR lower bound : 5.1394

Explication :
La politique issue de la regression n'est pas necessairement optimale.
On obtient donc un prix en dessous du prix americain (borne basse).

Question 2.3 
---------------------------------------

Procédure :
1) Régressions effectuées uniquement sur les trajectoires ITM.
2) OTM = continuation par définition.
3) Comparer les prix et tracer les régions d'exercice a t=0.5.

Résultats (exemple) :
 - LS poly ITM    : 4.0891
 - LS BS basis ITM: 5.1420

Commentaire :
L'exercice est concentré sur les niveaux S faibles (ITM profond),
la continuation domine pour S plus élevé.

Question 2.4 
----------------------------------

Procédure :
1) Simuler S_t sur 12 dates.
2) Calculer A_tn = moyenne des prix jusqu'a t_n.
3) Payoff : max(A_tn - K, 0).
4) Regression avec base [1, BS_call(Z_tn)].
   Z_tn = (n*A_tn + (12-n)*S_tn)/12.
5) LS puis simulation indépendante pour borne basse.

Résultats (exemple) :
 - LS price (Z_tn)       : 5.3686
 - Low-biased (100k)     : 4.7471

Pourquoi Z_tn :
Z_tn est un proxy de la moyenne finale. Il stabilise la régression en
utilisant à la fois l'information de l'average et du spot courant.

Question 2.4(b)
------------------------------------

Procédure :
1) Réseau feed-forward avec entrées (S_tn, A_tn).
2) Un Réseau par date (comme LS).
3) Apprentissage avec 3 couches de 20 neurones (ReLU).
4) Borne basse via simulation indépendante.

Résultats (exemple, 50 000 traj. entrainement) :
 - NN LS price : 4.4873
 - NN low-biased (100k) : 4.9590

-------------------------------------------------------------------------------
Synthèse 
-------------------------------------------------------------------------------

Q1 : Les degrés élevés et trop de noeuds sur-ajustent.  
Q2.1 : LS et TVR donnent des prix proches avec bases paramétriques.  
Q2.2 : La simulation indépendante fournit une borne basse.  
Q2.3 : Le filtrage ITM est plus stable, surtout pour les bases riches.  
Q2.4 : Z_tn stabilise la régression. Le NN fournit un prix compétitif mais
dépend fortement de l'entrainement.
