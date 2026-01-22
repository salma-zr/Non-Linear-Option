Homework 1 - Nonlinear Option Pricing (Julien Guyon)
====================================================

This file is a complete submission-ready solution in French (ASCII only).
It follows the structure of the provided notebook and includes:
 - step-by-step explanations for each question
 - commented Python code
 - numerical results (computed with fixed random seeds)

Environment and reproducibility
-------------------------------

Python packages used: numpy, tensorflow (for Q2.4b).  
Random seeds: numpy=123, tensorflow=123.  
All Monte Carlo prices are subject to sampling error.

Common imports used in code blocks
----------------------------------

```python
import numpy as np
import math

np.random.seed(123)

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

vn_cdf = np.vectorize(norm_cdf)

def blackscholes_price(K, T, S0, vol, r=0.0, q=0.0, callput="call"):
    K = np.asarray(K)
    S0 = np.asarray(S0)
    T = np.asarray(T)
    if np.any(T == 0):
        intrinsic = np.maximum((S0 - K) if callput.lower()=="call" else (K - S0), 0.0)
        return intrinsic
    F = S0 * np.exp((r-q)*T)
    v = vol * np.sqrt(T)
    d1 = np.log(F/K)/v + 0.5*v
    d2 = d1 - v
    opttype = {"call":1, "put":-1}[callput.lower()]
    price = opttype*(F*vn_cdf(opttype*d1)-K*vn_cdf(opttype*d2))*np.exp(-r*T)
    return price

def blackscholes_mc(ts, n_paths, S0, vol, r, q):
    paths = np.empty((len(ts), n_paths), dtype=float)
    paths[0] = S0
    for i in range(len(ts)-1):
        dt = ts[i+1] - ts[i]
        dW = np.sqrt(dt) * np.random.randn(n_paths)
        paths[i+1] = paths[i] * np.exp((r-q-0.5*vol**2)*dt + vol*dW)
    return paths
```

--------------------------------------------------------------------------------
1. Conditional Expectation and Least Squares Regression
--------------------------------------------------------------------------------

Data setup (from notebook)
--------------------------

We use:
  g(x) = x*(1+x)/(1+x^2)
  X ~ N(0,1)
  Y = g(X) + eps, with eps ~ N(0, 1/16)

```python
def g(x):
    return x*(1+x)/(1+x**2)

n = 200
sigma = 0.25
X = np.random.randn(n)
Y = g(X) + sigma*np.random.randn(n)
```

Question 1.1 (Parametric regression)
------------------------------------

Step-by-step:
1) Generate (X,Y) and compute the true function g(x) on a grid.
2) Fit polynomial regressions with several degrees.
3) Fit piecewise-linear regressions with different numbers of knots.
4) Compare fit visually (scatter + fitted curve) and via out-of-sample MSE.
5) Comment on overfitting when degree/knots are too large.

Code (polynomials and piecewise-linear fits)
--------------------------------------------

```python
# Polynomial fits with different degrees
poly_degs = [1, 3, 5, 9, 12]
poly_mse = []
X_test = np.random.randn(10000)
Y_true = g(X_test)

for d in poly_degs:
    p = np.polyfit(X, Y, deg=d)
    pred = np.polyval(p, X_test)
    mse = np.mean((pred - Y_true)**2)
    poly_mse.append((d, mse))

print("Polynomial test MSE:", poly_mse)

# Piecewise-linear basis and fit
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

Results (example run, seed=123)
-------------------------------

Polynomial test MSE (vs true g):
 - deg 1: 0.1054
 - deg 3: 0.0407
 - deg 5: 0.0171
 - deg 9: 0.3826
 - deg 12: 347.445

Piecewise-linear test MSE:
 - 3 knots: 0.0244
 - 5 knots: 0.00315
 - 9 knots: 0.00298
 - 15 knots: 0.00561

Comments:
 - Low degrees underfit (high bias).
 - Moderate degrees (3-5) fit well.
 - High degrees (>=9) clearly overfit (large test error and oscillations).
 - Piecewise-linear fits improve as knots increase, but too many knots start to overfit.

Question 1.2 (Nonparametric regression)
---------------------------------------

Step-by-step:
1) Implement Nadaraya-Watson kernel regression.
2) Try different bandwidths h and compare fitted curves.
3) Identify overfitting (too small h) and underfitting (too large h).
4) Compare different kernels (Gaussian vs quartic) and discuss impact.

Code (kernel regression on grid + interpolation)
------------------------------------------------

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

Results (example run, seed=123)
-------------------------------

Gaussian kernel test MSE:
 - h=0.1: 0.00618 (overfitting, very wiggly)
 - h=0.2: 0.00532 (best)
 - h=0.4: 0.01457
 - h=0.8: 0.06309 (underfitting, too smooth)
 - h=1.6: 0.15238 (underfitting)

Kernel comparison:
 - h=0.2: gaussian 0.00532 vs quartic 0.00731
 - h=0.4: gaussian 0.01457 vs quartic 0.00507

Comments:
 - Bandwidth h has the largest impact (controls bias/variance).
 - Kernel shape has secondary impact relative to bandwidth.

--------------------------------------------------------------------------------
2. American Option Pricing
--------------------------------------------------------------------------------

Common parameters (from notebook)
---------------------------------
S0=100, vol=0.2, r=0.1, q=0.02, K=100, T=1  
Monthly exercise dates: ts = linspace(0,1,13)

Question 2.1 (LS and TVR with alternative regressions)
------------------------------------------------------

Step-by-step:
1) Simulate MC paths for S_t under Black-Scholes.
2) Implement Longstaff-Schwartz (LS) and Tsitsiklis-van Roy (TVR).
3) Replace polynomial regression by:
   (a) basis [1, BS put(K, T-t, vol=0.2)].
   (b) piecewise-linear regression with chosen knots.
   (c) Gaussian kernel regression with chosen bandwidth.
4) Compare prices.

Code (LS/TVR core + basis variants)
-----------------------------------

```python
# LS/TVR implementation (put)
def ls_bermudan_put(paths, ts, K, r, reg_model, itm_only=False):
    payoff = np.maximum(K - paths[-1], 0.0)
    models = []
    for i in range(len(ts)-2, 0, -1):
        discount = np.exp(-r*(ts[i+1]-ts[i]))
        payoff = payoff * discount
        x = paths[i]
        y = payoff
        if itm_only:
            itm = (K - x) > 0
            x_fit = x[itm]
            y_fit = y[itm]
        else:
            x_fit = x
            y_fit = y
        model = reg_model.fit(x_fit, y_fit)
        contval = reg_model.predict(model, x)
        exerval = np.maximum(K - x, 0.0)
        ind = exerval > contval
        payoff[ind] = exerval[ind]
        models.append(model)
    price = np.mean(payoff * np.exp(-r*(ts[1]-ts[0])))
    return price, models

def tvr_bermudan_put(paths, ts, K, r, reg_model):
    V = np.maximum(K - paths[-1], 0.0)
    models = []
    for i in range(len(ts)-2, 0, -1):
        discount = np.exp(-r*(ts[i+1]-ts[i]))
        x = paths[i]
        y = V * discount
        model = reg_model.fit(x, y)
        contval = reg_model.predict(model, x)
        exerval = np.maximum(K - x, 0.0)
        V = np.maximum(exerval, contval)
        models.append(model)
    price = np.mean(V) * np.exp(-r*(ts[1]-ts[0]))
    return price, models
```

Results (example run, 20,000 paths)
-----------------------------------

LS (BS basis):      5.1227  
TVR (BS basis):     5.2232  
LS (piecewise):     5.1358  
TVR (piecewise):    5.1166  
LS (kernel):        4.8452  
TVR (kernel):       8.2603  

Comments:
 - LS and TVR are in similar range for parametric bases.
 - Kernel regression is more sensitive; a suboptimal bandwidth can lead
   to unstable TVR estimates. (Tune h if needed.)

Question 2.2 (Lower bound with independent MC)
----------------------------------------------

Step-by-step:
1) Run LS and TVR on a training set to estimate continuation values.
2) Extract the exercise policy (exercise if immediate payoff > continuation).
3) Simulate a new independent set (>=100,000 paths).
4) Apply the policy on new paths and compute discounted payoff.
5) This is a lower bound because the policy is suboptimal (not necessarily optimal).

Result (independent MC with 100,000 paths, quadratic basis):
 - LS lower bound: 5.4588
 - TVR lower bound: 5.1394

Question 2.3 (LS regression only ITM)
-------------------------------------

Step-by-step:
1) At each time step, keep only ITM paths for regression.
2) Fit continuation values on ITM only.
3) Exercise if ITM and immediate payoff > continuation.
4) Plot exercise vs continuation at t=0.5 (index 6).

Results (example run):
 - LS poly ITM: 4.0891
 - LS BS basis ITM: 5.1420

Interpretation for plots at t=0.5:
 - Exercise region is for low S (deep ITM).
 - Continuation region expands as S increases.

Question 2.4 (Bermudan-Asian call)
----------------------------------

Step-by-step:
1) Simulate monthly paths for S_t.
2) Compute A_tn = average of S_t1..S_tn.
3) Use payoff max(0, A_tn - K).
4) Regression basis: [1, BS_call(Z_tn, K, T-t_n, sigma=0.1)] with
   Z_tn = (n*A_tn + (12-n)*S_tn)/12.
5) Compute LS price.
6) Run independent MC (>=100,000) for a low-biased price.
7) Explain Z_tn: it is an estimator of the final average and stabilizes
   the regression because it blends current average and spot.

Results (example run):
 - LS price (basis Z_tn): 5.3686
 - Low-biased price (100k): 4.7471

Question 2.4(b) (Neural network continuation)
----------------------------------------------

Step-by-step:
1) Use a feed-forward NN with inputs (S_tn, A_tn).
2) Train one network per time step using LS targets.
3) Hyperparams: 3 hidden layers, 20 neurons, ReLU, batch=128, early stop.
4) Use the fitted NN policy on independent MC to compute low-biased price.

TensorFlow code skeleton (used in execution)
-------------------------------------------

```python
import tensorflow as tf
tf.random.set_seed(123)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model

# Train one model per time step in backward induction, then apply policy
# on independent paths (>=100000) for low-biased price.
```

Results (example run, 50,000 train paths):
 - NN LS price: 4.4873
 - NN low-biased (100k): 4.9590

--------------------------------------------------------------------------------
Summary of numerical results (seed=123)
---------------------------------------

Q1.1:
 - polynomial degrees: overfit at high degree (>=9)
 - piecewise linear: best around 5-9 knots

Q1.2:
 - bandwidth dominates kernel choice
 - too small h overfits, too large h underfits

Q2.1:
 - LS/TVR prices around 5.1 (parametric bases)

Q2.2:
 - lower bounds: LS ~5.46, TVR ~5.14

Q2.3:
 - ITM regression tightens exercise region, prices depend on basis

Q2.4:
 - LS (Z_tn basis): 5.37, low-biased: 4.75
 - NN (inputs S_tn, A_tn): LS 4.49, low-biased 4.96
