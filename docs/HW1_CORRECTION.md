## HW1 (used here as the provided “annale”) — Fully annotated correction

Source: `HW1-SorbonneUniversite.ipynb (1).json`.

### Convention and notation

- \((\Omega,\mathcal F,(\mathcal F_t)_{t\in[0,T]},\mathbb P)\) filtered probability space.
- Discount factor \(D_{t,s}=e^{-\int_t^s r_u\,du}\) (constant \(r\) in HW1 examples).
- For Bermudan exercise dates \(t_0<\dots<t_N\), the value satisfies the **Snell recursion**:
\[
V_{t_N}=F_{t_N},\qquad
V_{t_i}=\max\Big(F_{t_i},\ \mathbb E[D_{t_i,t_{i+1}}V_{t_{i+1}}\mid\mathcal F_{t_i}]\Big).
\]

---

## Question 1.1 — Parametric regression and overfitting

### Restatement

Using polynomial regression (vary degree) and piecewise-linear regression (vary number of knots), reproduce the scatter plot with fitted regression function and comment on overfitting vs underfitting.

### Solution (rigorous, with the key idea)

Let \(X\sim\mathcal N(0,1)\), \(Y=g(X)+\varepsilon\) with \(\varepsilon\perp X\), \(\mathbb E[\varepsilon]=0\).

#### Step 1 — Conditional expectation as an \(L^2\) projection

\[
f^\*(X):=\mathbb E[Y\mid X]
\]
is the unique (a.s.) minimizer of
\[
\min_{f\in L^2(\sigma(X))}\ \mathbb E[(Y-f(X))^2].
\]
Moreover, since \(Y=g(X)+\varepsilon\) and \(\varepsilon\) is independent with mean zero,
\[
\mathbb E[Y\mid X]=g(X).
\]

#### Step 2 — Parametric regression as projection on a finite-dimensional subspace

Pick basis functions \(\phi_0,\dots,\phi_m\) (polynomials; or piecewise-linear “hat” functions). You approximate \(g\) by
\[
f_\beta(x)=\sum_{j=0}^m \beta_j\phi_j(x),
\]
and choose \(\beta\) via empirical least squares (OLS):
\[
\hat\beta=\arg\min_\beta \frac1N\sum_{i=1}^N (y_i-f_\beta(x_i))^2.
\]

#### Step 3 — Bias–variance tradeoff explains under/overfitting

Write the *prediction risk* at a new point as
\[
\mathbb E[(Y-f_\beta(X))^2]
=\underbrace{\mathbb E[(g(X)-f_\beta(X))^2]}_{\text{approximation error (bias}^2\text{)}}+\underbrace{\mathbb E[\varepsilon^2]}_{\text{irreducible noise}}.
\]
With estimated \(\hat\beta\), you add **estimation variance**. Increasing model complexity (degree / knots) typically:
- decreases approximation error (more flexibility),
- increases estimation variance (more sensitivity to sample noise),
so out-of-sample error is U-shaped.

### Tools used / why they apply / assumptions

- **Tool**: conditional expectation as \(L^2\) projection.
  - **Why**: regression is exactly approximating \(\mathbb E[Y\mid X]\).
  - **Assumptions**: \(Y\in L^2\).
- **Tool**: OLS normal equations / least squares.
  - **Why**: parametric class is linear in coefficients.
  - **Assumptions**: design matrix full rank for uniqueness.

### Common traps

- Confusing “better in-sample fit” with “better predictor.”
- Using too high degree: oscillations (Runge-like behavior) near tails where data are sparse.
- Piecewise-linear with many knots: local interpolation of noise.

---

## Question 1.2 — Nonparametric kernel regression (bandwidth vs kernel)

### Restatement

Try different bandwidths \(h\) in Gaussian-kernel regression; compare fit and identify overfitting vs poor fit. Try a different kernel. Decide what matters most: \(h\) or \(K\).

### Solution

#### Step 1 — Nadaraya–Watson estimator

With kernel \(K\) and bandwidth \(h\),
\[
\hat g_h(x)=\frac{\sum_{i=1}^N K\!\left(\frac{x-x_i}{h}\right)y_i}{\sum_{i=1}^N K\!\left(\frac{x-x_i}{h}\right)}.
\]

#### Step 2 — Role of \(h\) (dominant effect)

- **Small \(h\)**: very local averaging ⇒ high variance ⇒ the curve tracks noise (**overfitting**).
- **Large \(h\)**: heavy smoothing ⇒ high bias ⇒ the curve misses curvature (**underfitting/poor fit**).

Asymptotically (for smooth \(g\)), the leading behavior is:
- Bias \(\sim O(h^2)\),
- Variance \(\sim O((Nh)^{-1})\).
So \(h\) is the main knob controlling bias/variance.

#### Step 3 — Role of kernel \(K\)

For standard smooth kernels (Gaussian, Epanechnikov, quartic), **kernel choice affects constants** in the bias/variance expansions, but **bandwidth choice dominates** practical behavior.

### Tools used / assumptions

- **Tool**: kernel smoothing as local averaging estimator of \(\mathbb E[Y\mid X=x]\).
- **Assumptions**: i.i.d. sample, \(g\) reasonably smooth for bias/variance heuristics.

### Common traps

- Evaluating fit quality on the training sample only (will favor tiny \(h\)).
- Ignoring boundary/tail regions where effective sample size is smaller.

---

## Question 2.1 — LSM and TVR with alternative regression methods

### Restatement

Modify Longstaff–Schwartz (LSM) and Tsitsiklis–Van Roy (TVR) Bermudan put pricing code to use:

- (a) basis \(\{1,\ P^{BS}_{\bar\sigma}(t,S_t;K,T-t)\}\),
- (b) piecewise-linear regression (choose knots),
- (c) Gaussian kernel regression (choose bandwidth).

### General solution pattern (applies to all (a)(b)(c))

You are implementing the Snell recursion at dates \(t_i\):
\[
V_{t_i}=\max\big(F_{t_i},\ C_{t_i}\big),\qquad
C_{t_i}:=\mathbb E[D_{t_i,t_{i+1}}V_{t_{i+1}}\mid S_{t_i}],
\]
and approximating \(C_{t_i}\) by regression on a function class.

#### LSM (policy-based) vs TVR (value-function-based): what differs

- **LSM**: regress *realized discounted cashflows* from continuation on features of \(S_{t_i}\); use \(\hat C_{t_i}\) only to decide exercise; once exercised, path cashflow is fixed.
- **TVR**: regress a proxy of the value function backward and apply \(V=\max(F,\hat C)\) directly in the recursion.

**Important bias fact**:
- If you **evaluate the policy out-of-sample** (Question 2.2), you obtain a **valid lower bound**.
- TVR recursion tends to be **upward biased** because applying \(\max(\cdot,\hat C)\) with noisy \(\hat C\) introduces Jensen/“max” bias.

### (a) Black–Scholes put price basis

At time \(t_i\), define features
\[
\phi_0(S_{t_i})\equiv 1,\qquad
\phi_1(S_{t_i}) := P^{BS}\big(S_{t_i},K,T-t_i,\bar\sigma,r,q\big).
\]
Then estimate continuation by OLS:
\[
\hat C_{t_i}(S_{t_i})=\hat\beta_0\phi_0(S_{t_i})+\hat\beta_1\phi_1(S_{t_i}),
\]
where \(\hat\beta\) solves least squares against the discounted next-step value (TVR) or discounted future realized payoff (LSM).

**Intuition**: the European put price is a strong nonlinear regressor correlated with continuation value.

### (b) Piecewise-linear regression

Choose knots \(k_1<\dots<k_m\) (e.g., empirical quantiles of \(S_{t_i}\) on in-the-money paths).
Use hat functions \(\psi_j\) (piecewise-linear basis), and regress
\[
\hat C_{t_i}(s)=\sum_{j=0}^m \hat\beta_j\psi_j(s)
\]
with \(\psi_0\equiv 1\).

**Knot choice** is a bias/variance tradeoff: too many knots overfit; too few underfit.

### (c) Gaussian kernel regression

Use Nadaraya–Watson (or local linear) estimator for continuation:
\[
\hat C_{t_i}(s)=\frac{\sum_{n=1}^N K\!\left(\frac{s-S_{t_i}^{(n)}}{h}\right)\,Y^{(n)}}{\sum_{n=1}^N K\!\left(\frac{s-S_{t_i}^{(n)}}{h}\right)},
\]
where \(Y^{(n)}\) is the discounted continuation target on path \(n\).

Bandwidth \(h\) controls smoothness; in 1D it is feasible, but in higher dimension it suffers from the curse of dimensionality.

### Tools used / why / assumptions

- **Tool**: Snell envelope recursion (discrete time).
  - **Why**: Bermudan approximation of American option.
  - **Assumptions**: integrability of discounted payoff; Markov state \(S_{t_i}\) (Black–Scholes).
- **Tool**: regression = projection of conditional expectation on a function class.
  - **Why**: continuation value is a conditional expectation.
  - **Assumptions**: sufficient paths; stable design (avoid collinearity).

### Typical mistakes

- Using in-the-money restriction incorrectly (see Q2.3).
- Forgetting discount factor \(e^{-r\Delta t}\).
- Using the same paths to both fit and evaluate and claiming “lower bound” (needs out-of-sample).

---

## Question 2.2 — Independent Monte Carlo lower bound and explanation

### Restatement

Using the estimated continuation function from **both** TVR and LSM as an exercise policy, run an independent Monte Carlo simulation (≥ 100000 paths), estimate the American put price, and explain why it is a lower bound.

### Solution

#### Step 1 — Build a stopping rule from the estimated continuation value

On the regression (training) stage you obtain \(\hat C_{t_i}(s)\).
Define the exercise rule:
\[
\hat\tau := \inf\{t_i:\ F_{t_i}\ge \hat C_{t_i}(S_{t_i})\}
\]
(with \(\hat\tau=T\) if never exercised).

#### Step 2 — Evaluate the policy on independent paths

Simulate fresh paths \((S_{t_i})\) independent of the training sample and compute
\[
\widehat V_0^{\,\text{policy}}=\frac1M\sum_{m=1}^M D_{0,\hat\tau^{(m)}}F_{\hat\tau^{(m)}}.
\]

#### Step 3 — Why it is a lower bound

Because \(\hat\tau\in\mathcal T_{0,T}\) is an admissible stopping time (exercise strategy),
\[
\mathbb E[D_{0,\hat\tau}F_{\hat\tau}] \le \sup_{\tau\in\mathcal T_{0,T}}\mathbb E[D_{0,\tau}F_\tau]=V_0.
\]
The independent Monte Carlo estimate is (approximately) unbiased for \(\mathbb E[D_{0,\hat\tau}F_{\hat\tau}]\), hence it estimates a quantity \(\le V_0\).

### Tools used / assumptions

- **Tool**: definition of the American price as a supremum over stopping times.
- **Tool**: Monte Carlo law of large numbers (independent evaluation).
- **Assumptions**: the policy depends only on current/past information (measurable), and paths are simulated under the pricing measure used in the recursion.

### Common traps

- Calling an in-sample estimate a “lower bound” (max + regression noise can create upward bias).
- Mixing measures (simulating under \(\mathbb P\) while discounting as if under \(\mathbb Q\)).

---

## Question 2.3 — Regress only in-the-money (ITM) and plot regions at \(t=0.5\)

### Restatement

Modify LSM to run regression only on ITM paths. Use:
- (a) quadratic polynomial basis,
- (b) basis from Q2.1(a).
Plot exercise/continuation regions at \(t=0.5\), comment.

### Solution

#### Step 1 — ITM restriction rationale

If \(F_{t_i}=0\) (out of the money), exercising is dominated by continuing, so continuation estimation is not needed there.

#### Step 2 — ITM regression step

At each \(t_i\), restrict the regression sample to indices
\[
\mathcal I_i:=\{n:\ F_{t_i}^{(n)}>0\}.
\]
Fit \(\hat C_{t_i}\) using only \(\{S_{t_i}^{(n)}:n\in\mathcal I_i\}\) and targets \(\{Y^{(n)}:n\in\mathcal I_i\}\).

Then define exercise decision:
- if \(F_{t_i}=0\): continue,
- else: exercise if \(F_{t_i}\ge \hat C_{t_i}(S_{t_i})\).

#### Step 3 — Plotting exercise vs continuation region at \(t=0.5\)

At \(t=0.5\) (i.e., \(t_i=0.5\) on the grid):
- compute \(s\mapsto F(s)\) and \(s\mapsto \hat C_{t_i}(s)\),
- **exercise region**: \(\{s: F(s)\ge \hat C_{t_i}(s)\}\),
- **continuation region**: complement.

**Expected qualitative result for an American put**: exercise for sufficiently low \(S\) (deep ITM), continue for higher \(S\); the boundary is decreasing in time-to-maturity.

### Tools used

- **Tool**: Snell envelope recursion + regression approximation of conditional expectation.

### Common traps

- Regressing on all paths but setting targets to zero OTM (distorts conditional expectation).
- Plotting region using in-sample noisy estimates without smoothing (gives scattered boundary).

---

## Question 2.4 — Bermudan Asian call (basis functions and neural net)

### Restatement

Price a Bermudan-Asian call exercisable monthly with payoff \((A_{t_n}-K)^+\), \(A_{t_n}=\frac1n\sum_{i\le n}S_{t_i}\).

- (a) LSM with basis including constant and a proxy European call price using \(Z_{t_n}=\frac{nA_{t_n}+(12-n)S_{t_n}}{12}\) (and explain \(Z_{t_n}\)); then independent MC lower bound.
- (b) Use a feed-forward NN taking \((S_{t_n},A_{t_n})\) as inputs to estimate continuation; then independent MC lower bound.

### Solution

#### Step 1 — Markov state choice (key modeling step)

Asian payoff is path-dependent, so the minimal Markov state is typically \((S_t,A_t)\) (or \((S_t,\sum S_{t_i})\) on a grid).
That is why part (b) uses \((S_{t_n},A_{t_n})\) as NN inputs.

#### Step 2 — Why the proxy \(Z_{t_n}\) helps (part (a))

\[
Z_{t_n}=\frac{nA_{t_n}+(12-n)S_{t_n}}{12}
\]
is a **proxy for the full 12-month average** mixing what is already known (\(A_{t_n}\)) with an approximation of the remaining contribution using \(S_{t_n}\).

Conceptually, under GBM the future average (conditioned on current information) is strongly correlated with current \(S_{t_n}\). Using \(Z_{t_n}\) produces a regressor closer to the exercise payoff at maturity, improving continuation estimation.

#### Step 3 — LSM with basis functions (part (a))

At each \(t_n\), define features (as instructed):
- constant \(1\),
- European call price \(C^{BS}(Z_{t_n},K,T-t_n,\bar\sigma,r,q)\) with \(\bar\sigma=0.1\),
- (often also include \(Z_{t_n}\) itself if implementing “and the spot value \(Z_{t_n}\)” literally as an additional basis).

Regress discounted continuation cashflows on these features to obtain \(\hat C_{t_n}(S_{t_n},A_{t_n})\) (through the proxy).

Then, as in Q2.2, use an **independent simulation** to estimate the policy value, giving a lower bound.

#### Step 4 — Neural network continuation approximation (part (b))

You approximate the conditional expectation
\[
C_{t_n}(S_{t_n},A_{t_n})=\mathbb E[D_{t_n,t_{n+1}}V_{t_{n+1}}\mid S_{t_n},A_{t_n}]
\]
by a neural network \(N_\theta(S_{t_n},A_{t_n})\) trained to minimize MSE to discounted realized continuation targets.

Then exercise rule:
\[
\text{exercise at }t_n \text{ if } (A_{t_n}-K)^+\ \ge\ N_\theta(S_{t_n},A_{t_n}).
\]

Finally, evaluate the induced stopping policy on independent paths to obtain a lower bound.

### Tools used / assumptions

- **Tool**: dynamic programming / Snell envelope on the Bermudan grid.
- **Tool**: conditional expectation approximation by regression / NN (universal approximation as intuition).
- **Assumptions**: sufficient training data; stable optimization; independence for lower-bound evaluation.

### Typical mistakes

- Treating \(A_{t_n}\) as Markov alone (it is not; \(S_{t_n}\) matters for future evolution).
- Using \((S_{t_n})\) only (insufficient state) and expecting correct continuation values.
- Reporting the in-sample recursion value as “the price” without out-of-sample validation.
