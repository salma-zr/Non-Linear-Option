# Nonlinear Option Pricing --- Complete Study Guide
## Master 2 Probabilites et Finance, Sorbonne Universite
### Based on Julien Guyon's Lecture Notes and Homework I

---

# PART 1 --- COURSE STRUCTURING

## 1.1 Logical Architecture of the Course

The course builds a tower of increasing nonlinearity, starting from the classical linear world and progressively introducing tools to handle genuinely nonlinear problems.

### Layer 0: Foundations (Prerequisites)
- Brownian motion, Ito calculus, SDEs
- Black-Scholes pricing PDE (linear)
- Feynman-Kac representation theorem
- Self-financing portfolios, no-arbitrage, equivalent martingale measures (ELMM)

### Layer 1: Early Exercise --- Optimal Stopping
- American / Bermudan option pricing
- Snell envelope (discrete time)
- Variational inequality (continuous time, Markovian case)
- Monte Carlo methods for Bermudan options:
  - Tsitsiklis-Van Roy (TVR) algorithm
  - Longstaff-Schwartz (LS) algorithm
- Machine learning techniques for conditional expectation estimation:
  - Parametric regression (polynomial, piecewise-linear, ridge)
  - Nonparametric regression (Nadaraya-Watson kernel regression, local linear regression)
  - Neural networks (feedforward)
- Jensen's bias and lower/upper bound pricing
- Primal-Dual methods (Rogers / Haugh-Kogan)
- Doob-Meyer decomposition and the Broadie-Andersen dual algorithm

### Layer 2: Option Pricing Theory in a Nutshell
- Superreplication paradigm
- Complete vs. incomplete markets
- ELMM characterization of no-arbitrage
- Buyer's and seller's prices
- Attainable payoffs and market completeness
- Pricing in practice: picking a particular ELMM

### Layer 3: Stochastic Representation of Solutions of Linear PDEs
- The Cauchy problem and Feynman-Kac
- Source terms (continuous cash flows)
- Path-dependent options

### Layer 4: Nonlinear Extensions (Covered in later lectures)
- Uncertain volatility, uncertain mortality
- Different rates for borrowing/lending
- BSDEs (Backward Stochastic Differential Equations)
- McKean SDEs and the particle method
- Branching diffusions
- CVA, transaction costs, super-replication under constraints
- Viscosity solutions

---

## 1.2 Key Definitions

### Definition 1: European Option Value
$$V^E_t = \mathbb{E}^Q[D_{t,T} F_T \mid \mathcal{F}_t]$$
where $D_{t,T} = e^{-\int_t^T r_s\,ds}$ is the discount factor.

### Definition 2: American Option Value
$$V^A_t = \sup_{\tau \in \mathcal{T}_{t,T}} \mathbb{E}^Q[D_{t,\tau} F_\tau \mid \mathcal{F}_t]$$
The supremum is over all stopping times $\tau \in [t,T]$, conditional on no prior exercise.

### Definition 3: Bermudan Option Value
$$V^B_t = \sup_{\tau \in \mathcal{T}_D} \mathbb{E}^Q[D_{t,\tau} F_\tau \mid \mathcal{F}_t]$$
where $\mathcal{T}_D$ is the set of stopping times taking values in $\{t_1, \ldots, t_N\} \cap [t,T]$.

### Definition 4: Snell Envelope (Discrete Time)
Given a process $\{Z_k\}_{1 \le k \le N}$, the Snell envelope is:
$$U_k = \sup_{\tau \in \mathcal{T}_{k,N}} \mathbb{E}[Z_\tau \mid \mathcal{F}_k]$$

**Recursive construction:**
$$U_N = Z_N, \quad U_k = \max\{Z_k, \mathbb{E}[U_{k+1} \mid \mathcal{F}_k]\}, \quad k = 0, \ldots, N-1$$

**Key property:** The Snell envelope is the *smallest supermartingale* dominating $\{Z_k\}$.

### Definition 5: Optimal Stopping Time
$$\tau^*_k = \inf\{n \ge k : Z_n = U_n\}$$
The stopped process $\{U_{n \wedge \tau^*_k}\}_{n \ge k}$ is a martingale.

### Definition 6: Continuation Value
$$C_{t_i} = \mathbb{E}^Q[D_{t_i, t_{i+1}} V_{t_{i+1}} \mid \mathcal{F}_{t_i}]$$
This is the expected discounted future value if one does NOT exercise now.

### Definition 7: Conditional Expectation as Best Predictor
If $Y \in L^2$, then $f^*(X) = \mathbb{E}[Y|X]$ minimizes $\mathbb{E}[(Y - f(X))^2]$ over all $f$ such that $f(X) \in L^2$.

### Definition 8: Self-Financing Portfolio
A portfolio $\pi_t = \sum_{i=0}^m \Delta^i_t X^i_t$ is self-financing if $d\pi_t = \sum_{i=0}^m \Delta^i_t\,dX^i_t$ (no cash injection or removal).

### Definition 9: Admissible Portfolio
$(\Delta_t)$ defines an admissible portfolio if $\tilde\pi_t$ is bounded from below for all $t$, $\mathbb{P}^{hist}$-a.s.

### Definition 10: Equivalent Local Martingale Measure (ELMM)
A measure $Q \sim \mathbb{P}^{hist}$ such that, for all assets $X^i$, the discounted price process $\{\tilde{X}^i_t\}$ is a $Q$-local martingale.

### Definition 11: Buyer's Super-replication Price
$$\mathcal{B}_t(F_T) = \sup\{z \in \mathcal{F}_t : \exists \text{ admissible } \Delta \text{ s.t. } -D_{0,t}z + \int_t^T \Delta_s \cdot d\tilde{X}_s + D_{0,T}F_T \ge 0 \text{ a.s.}\}$$

### Definition 12: Complete Market
A market is complete when every payoff is attainable at time 0.

**Theorem:** A market is complete $\iff$ there exists a *unique* ELMM.

---

## 1.3 Core Theorems

### Theorem 1: Variational Inequality (American Options, Markovian Case)
$$\max(\partial_t u + \mathcal{L}u - r u,\; g(x) - u(t,x)) = 0, \quad u(T,x) = g(x)$$
where $\mathcal{J} = \partial_t + \mathcal{L} - r$.

**Intuition:** At each point in space-time, either (a) you are in the *continuation region* where $u > g$ and the PDE $\mathcal{J}u = 0$ holds (the option behaves like a European option locally), or (b) you are in the *exercise region* where $u = g$ and $\mathcal{J}u \le 0$ (it is optimal to exercise immediately).

### Theorem 2: Recursive Construction of Snell Envelope
$$U_k = \sup_{\tau \in \mathcal{T}_{k,N}} \mathbb{E}[Z_\tau \mid \mathcal{F}_k] \iff \begin{cases} U_N = Z_N \\ U_k = \max\{Z_k, \mathbb{E}[U_{k+1} \mid \mathcal{F}_k]\} \end{cases}$$

**Why it matters:** This is the *engine* of all backward-induction Monte Carlo algorithms. Every Bermudan pricing algorithm (TVR, LS, dual methods) is built on this recursion.

### Theorem 3: Doob Decomposition (Discrete Time)
Any adapted integrable process $U_n$ admits a unique decomposition $U_n = M_n - A_n$ where $M_n$ is a martingale and $A_n$ is predictable with $A_0 = 0$. Moreover, $U_n$ is a supermartingale iff $A_n$ is increasing.

### Theorem 4: Primal-Dual Theorem (Rogers / Haugh-Kogan)
$$\sup_{\tau \in \mathcal{T}_{t,T}} \mathbb{E}^Q[D_{t,\tau} F_\tau \mid \mathcal{F}_t] = \inf_{M \in \mathcal{M}_{t,0}} \mathbb{E}^Q\left[\sup_{t \le s \le T}(D_{t,s}F_s - M_s) \mid \mathcal{F}_t\right]$$
where $\mathcal{M}_{t,0}$ is the set of right-continuous martingales with $M_t = 0$.

**Intuition:** The primal gives a *lower bound* (any exercise strategy underperforms the optimal). The dual gives an *upper bound* (any martingale "hedge" overestimates the option value). The optimal martingale $M^*$ is the martingale part of the Doob-Meyer decomposition of the Snell envelope.

### Theorem 5: Super-replication Price (El Karoui-Quenez, Kramkov)
$$\mathcal{B}_t(F_T) = \inf_{Q \in \text{ELMM}} \mathbb{E}^Q[D_{t,T}F_T \mid \mathcal{F}_t], \quad \mathcal{S}_t(F_T) = \sup_{Q \in \text{ELMM}} \mathbb{E}^Q[D_{t,T}F_T \mid \mathcal{F}_t]$$

### Theorem 6: Feynman-Kac
Under appropriate regularity conditions, the solution $u$ of the linear parabolic PDE
$$\partial_t u + \mathcal{L}u - r u + f = 0, \quad u(T,x) = g(x)$$
admits the stochastic representation:
$$u(t,x) = \mathbb{E}^Q\left[\int_t^T e^{-\int_t^s r\,du}\,f(s,X_s)\,ds + e^{-\int_t^T r\,ds}\,g(X_T) \mid X_t = x\right]$$

### Theorem 7: Universal Approximation Theorem (Neural Networks)
Any continuous function on $[0,1]^n$ can be approximated to arbitrary accuracy by a neural network with a single hidden layer, provided the hidden layer is sufficiently wide.

---

## 1.4 Central Mathematical Tools and Their Intuitive Roles

| Tool | Role in Nonlinear Option Pricing |
|------|----------------------------------|
| **Snell Envelope** | The fundamental object for optimal stopping. The discounted Bermudan option value process IS the Snell envelope of the discounted payoff. |
| **Doob-Meyer Decomposition** | Splits a supermartingale (the Snell envelope) into a martingale + decreasing process. The martingale part is the key ingredient for dual upper bounds. |
| **Dynamic Programming / Backward Induction** | The Snell envelope recursion $U_k = \max\{Z_k, \mathbb{E}[U_{k+1}\mid\mathcal{F}_k]\}$ IS dynamic programming. This is the conceptual backbone of ALL Bermudan pricing. |
| **Conditional Expectation Estimation** | The bottleneck of Monte Carlo Bermudan pricing: we need to estimate $\mathbb{E}[Y\mid X=x]$ from simulated data. Parametric regression, kernel methods, and neural networks are all tools for this. |
| **Jensen's Inequality / Bias Analysis** | $\max(x,a)$ is convex $\Rightarrow$ $\mathbb{E}[\max(a, \bar{X}_n)] \ge \max(a, \mathbb{E}[X])$. This creates *high bias* when the continuation value estimate has variance. Understanding this is crucial for interpreting algorithm outputs. |
| **Primal-Dual Methods** | Give rigorous *lower bounds* (from exercise strategies) and *upper bounds* (from martingale hedges). A tight duality gap = good exercise policy. |
| **Feynman-Kac** | The bridge between PDEs and SDEs. Allows us to price options either by solving PDEs (finite differences) or by Monte Carlo simulation. |
| **Super-replication** | In incomplete markets, the price is not unique. Super-replication gives bounds but they are typically too wide for practical use. |
| **Variational Inequality** | The PDE formulation of American option pricing. Combines the PDE in the continuation region with the constraint $u \ge g$ in the exercise region. |

---

## 1.5 Deep Ideas vs. Technical Details

### Conceptually Important (Deep Ideas)
1. **The Snell envelope is everything:** All Bermudan/American pricing reduces to computing the Snell envelope. The recursive formula $U_k = \max\{Z_k, \mathbb{E}[U_{k+1}\mid\mathcal{F}_k]\}$ is the single most important equation in the course.

2. **Conditional expectation estimation is THE computational challenge:** In high dimensions, we cannot solve PDEs (curse of dimensionality). We must use Monte Carlo. But Monte Carlo requires estimating conditional expectations. The entire ML toolkit (regression, kernels, neural networks) enters through this door.

3. **Primal = lower bound, Dual = upper bound:** Any exercise strategy gives a lower bound. Any martingale gives an upper bound. This is elegant and practically powerful.

4. **The discounted value process of a fairly-priced derivative is a martingale** (under the risk-neutral measure, in a complete market). This single fact generates all of Black-Scholes theory.

5. **Nonlinearity arises from optimization:** The $\max$ in the Snell envelope recursion is what makes American option pricing nonlinear. Similarly, uncertain volatility, different borrowing/lending rates, etc., all introduce optimization or worst-case operators that break linearity.

### Technical Details (Important but secondary)
- Specific finite difference schemes (explicit, implicit, Crank-Nicholson)
- Silverman's rule of thumb for bandwidth selection
- Ridge regression regularization formulas
- Specific neural network architectures and training details
- The precise form of the Heston model dynamics
- Default risk modeling with Poisson intensity

---

# PART 2 --- HOMEWORK CORRECTION (Homework I)

## Section 1: Conditional Expectation and Least Square Regression

### Setup
- $g(x) = x\frac{1+x}{1+x^2}$, $X \sim \mathcal{N}(0,1)$, $Y = g(X) + \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, 1/16)$ independent of $X$.
- $\mathbb{E}[Y|X] = g(X)$ (since $\varepsilon \perp X$ and $\mathbb{E}[\varepsilon] = 0$).

### Question 1.1: Overfitting in Parametric Regression

**Restatement:** Experiment with different polynomial degrees and numbers of piecewise-linear knots. Observe and comment on overfitting.

**Rigorous Solution:**

**Tool used:** Parametric regression (least squares), specifically polynomial regression via `numpy.polyfit` and piecewise-linear regression via `numpy.linalg.lstsq`.

**Why it applies:** The conditional expectation $\mathbb{E}[Y|X]$ minimizes the mean squared error among all functions of $X$. When we restrict to a finite-dimensional function space (polynomials of degree $d$, or piecewise-linear with $n$ knots), we get the best approximation within that space.

**Key observations:**
1. **Low degree polynomials (deg 1-2):** Underfitting. The model is too rigid to capture the nonlinear shape of $g(x)$.
2. **Moderate degree polynomials (deg 3-5):** Good fit. The model captures the essential shape without overfitting.
3. **High degree polynomials (deg 10-20):** Overfitting. Wild oscillations appear, especially near the boundaries of the data. The polynomial tries to interpolate individual noise points.

Similarly for piecewise-linear:
1. **Few knots (2-3):** Underfitting.
2. **Moderate knots (5-10):** Good fit.
3. **Many knots (50+):** Overfitting --- the regression curve becomes jagged and follows noise.

**Underlying theorem:** The bias-variance tradeoff. More parameters $\Rightarrow$ lower bias (more flexible) but higher variance (more sensitive to noise). The optimal complexity balances these two.

**Common traps:**
- Forgetting that polynomial regression is numerically unstable for high degrees (the Vandermonde matrix becomes ill-conditioned).
- Not normalizing variables before regression.
- Confusing in-sample fit quality with out-of-sample prediction quality.

---

### Question 1.2: Nonparametric Regression --- Bandwidth and Kernel Choice

**Restatement:** (a) Try different bandwidths in Nadaraya-Watson and local linear regression. When does overfitting/poor fit occur? (b) Try a different kernel (quartic). Which matters more: bandwidth $h$ or kernel $K$?

**Rigorous Solution:**

**Tools used:** Nadaraya-Watson kernel regression and local linear regression.

**Key formula (Nadaraya-Watson):**
$$\hat{f}(x) = \frac{\sum_{i=1}^N K_h(x - x_i) y_i}{\sum_{i=1}^N K_h(x - x_i)}$$

**(a) Bandwidth effects:**
- **Very small $h$ (e.g., $h = 0.05$):** Overfitting. Each prediction point is influenced only by its nearest neighbors. The curve becomes extremely jagged.
- **Moderate $h$ (Silverman's rule: $h \approx 1.06\hat\sigma N^{-1/5}$):** Good fit. Smooth approximation of $g(x)$.
- **Very large $h$ (e.g., $h = 5$):** Underfitting. All data points contribute almost equally. The regression essentially becomes a constant (the sample mean of $Y$).

**(b) Kernel choice:**

Using the quartic kernel $K(x) = (x+1)^2(1-x)^2$ for $|x| \le 1$ (and 0 elsewhere) instead of the Gaussian kernel:

**Result:** The bandwidth $h$ has *far more* impact than the kernel $K$. This is a well-known result in nonparametric statistics. Different kernels with the same bandwidth produce very similar regression curves. But changing the bandwidth dramatically affects the result.

**Why:** The kernel mainly determines the *shape* of the weighting, but the bandwidth determines the *scale* --- how many data points effectively contribute to each local estimate. The scale is what controls the bias-variance tradeoff.

**Assumptions required:**
- $K \ge 0$, $\int K(x)\,dx = 1$ (kernel is a density)
- Sufficient data density in the region of estimation
- Smoothness of the true regression function

**Common trap:** Thinking that kernel choice is very important. In practice, bandwidth selection is *the* critical decision.

---

## Section 2: American Option Pricing

### Setup
- Black-Scholes model: $S_0 = 100$, $\sigma = 0.2$, $r = 0.1$, $q = 0.02$, $K = 100$, $T = 1$.
- Bermudan put with monthly exercises: $t_1 = 1/12, t_2 = 2/12, \ldots, t_{12} = 1$.
- Payoff: $F(S) = (K - S)^+$.

### Question 2.1: Alternative Regression Methods for LS and TVR

**Restatement:** Adapt Longstaff-Schwartz and TVR to use:
(a) Black-Scholes put prices as basis functions
(b) Piecewise linear regression
(c) Gaussian kernel regression

**Rigorous Solution:**

**General structure recall:** Both algorithms estimate the continuation value $C_{t_i} = \mathbb{E}^Q[D_{t_i,t_{i+1}} V_{t_{i+1}} | \mathcal{F}_{t_i}]$ at each exercise date via regression, then compare it with the exercise value.

#### (a) Black-Scholes Put Prices as Basis Functions

**Idea:** Use as basis functions at time $t_i$:
- $f_1(S) = 1$ (constant)
- $f_2(S) = \text{BS\_Put}(K, T-t_i, S, \bar\sigma, r, q)$ where $\bar\sigma = 0.2$

**Why this is clever:** A BS put price already captures most of the shape of the continuation value. The regression only needs to find the right linear combination, making it very robust even with few paths.

**Implementation (Longstaff-Schwartz):**
```python
payoff = np.maximum(K - paths[-1], 0)
for i in range(len(ts)-2, 0, -1):
    discount = np.exp(-r * (ts[i+1] - ts[i]))
    payoff = payoff * discount
    # Build design matrix with BS put prices
    bs_puts = blackscholes_price(K, T-ts[i], paths[i], vol, r, q, 'put')
    A = np.column_stack([np.ones(n_paths), bs_puts])
    beta = np.linalg.lstsq(A, payoff, rcond=None)[0]
    contval = A @ beta
    exerval = np.maximum(K - paths[i], 0)
    ind = exerval > contval
    payoff[ind] = exerval[ind]
price = np.mean(payoff * np.exp(-r * (ts[1] - ts[0])))
```

#### (b) Piecewise Linear Regression

**Implementation:** Use the `pwlin_fit` function with knots chosen at quantiles of the stock price distribution. A good choice is 5-10 knots spread across the 2.5th to 97.5th percentiles of $S_{t_i}$.

#### (c) Gaussian Kernel Regression

**Implementation:** Use the `kern_reg` (NW) or `ll_reg` (local linear) functions with Silverman's bandwidth. Evaluate at a grid of points, then interpolate.

**Caveat for kernel regression in LS/TVR:** Kernel regression is computationally expensive with many paths. The trick from the lecture notes is to evaluate at a grid of points and then interpolate using `scipy.interpolate.interp1d`.

**Tool used:** Conditional expectation estimation via parametric/nonparametric regression.
**Why it applies:** The Snell envelope recursion requires computing $\mathbb{E}[U_{k+1}|\mathcal{F}_k]$ from simulated data.
**Type of problem:** Optimal stopping / Bermudan option pricing.

---

### Question 2.2: Lower Bound Pricing via Independent Monte Carlo

**Restatement:** Use the exercise policy from LS/TVR (with quadratic polynomial basis) to estimate the American put price via an independent simulation. Explain why this is a lower bound. Show code with $\ge$ 100,000 paths.

**Rigorous Solution:**

**Recall the general result:** For any stopping time $\tau \in \mathcal{T}_{0,T}$:
$$\mathbb{E}^Q[D_{0,\tau} F_\tau] \le u_0 = \sup_{\tau' \in \mathcal{T}_{0,T}} \mathbb{E}^Q[D_{0,\tau'} F_{\tau'}]$$

Any *suboptimal* exercise strategy produces a value $\le$ the true American option value. Therefore:

**Step 1:** Run LS (or TVR) on $p_1$ paths to obtain an estimate of the optimal exercise policy:
$$\hat\tau_1 = \inf\{t_j : \hat{C}_{t_j} \le F_{t_j}\}$$

**Step 2:** Simulate $p_2$ *new, independent* paths. On each path, apply the exercise rule $\hat\tau_1$ and compute the discounted payoff.

**Step 3:** Average over the $p_2$ paths.

**Why this is a lower bound:**
1. The exercise policy $\hat\tau_1$ is almost certainly *not* the true optimal policy (it was estimated from finite data with regression error).
2. A suboptimal exercise policy always gives a value $\le$ the true supremum.
3. The second-step simulation uses *independent* paths, so there is no Jensen's bias from the $\max$ operation.

**Key insight:** In the first step (LS/TVR alone), the estimated price has *undetermined bias* because the regression errors interact with the $\max$ operation (Jensen's bias). The two-step procedure removes this: Step 1 determines the *policy*, Step 2 *evaluates* it on fresh data.

```python
# Step 1: Determine exercise policy using LS with p1 paths
p1 = 50000
paths1 = blackscholes_mc(ts, p1, S0, vol, r, q)
# ... (run LS backward induction to get regression coefficients at each time step)
# Store the regression coefficients for each exercise date

# Step 2: Evaluate on p2 independent paths  
p2 = 100000
paths2 = blackscholes_mc(ts, p2, S0, vol, r, q)
# For each path, find the first exercise date where exercise value > continuation value
# Compute discounted payoff and average
```

**Common trap:** Using the *same* paths for both training and evaluation. This reintroduces the Jensen's bias. The paths MUST be independent.

---

### Question 2.3: In-the-Money Regression (Longstaff-Schwartz Improvement)

**Restatement:** Modify LS to regress only on paths where the option is in the money. For out-of-the-money paths, always continue. Show results with (a) quadratic polynomial and (b) BS put price basis functions. Plot exercise/continuation regions at $t = 0.5$.

**Rigorous Solution:**

**Why this improvement works:** When the option is out of the money (exercise value = 0), it is *always* better to continue (continuation value $\ge 0$). So we don't need to estimate the continuation value there. By restricting regression to in-the-money paths, we:
1. Reduce noise in the regression (no near-zero observations)
2. Focus the regression's capacity on the region that actually matters for the exercise decision

**Modified LS algorithm:**
```python
payoff = np.maximum(K - paths[-1], 0)
for i in range(len(ts)-2, 0, -1):
    discount = np.exp(-r * (ts[i+1] - ts[i]))
    payoff = payoff * discount
    exerval = np.maximum(K - paths[i], 0)
    # Only regress on in-the-money paths
    itm = exerval > 0
    if np.sum(itm) > 0:
        # Regression only on ITM paths
        p = np.polyfit(paths[i, itm], payoff[itm], deg=2)
        contval = np.polyval(p, paths[i])
        # Exercise only where ITM AND exercise > continuation
        ind = itm & (exerval > contval)
        payoff[ind] = exerval[ind]
price = np.mean(payoff * np.exp(-r * (ts[1] - ts[0])))
```

**Exercise/continuation regions at $t = 0.5$:**
- The exercise boundary is the stock price $S^*$ below which it is optimal to exercise.
- For the American put: exercise when $S < S^*$ (deeply in the money).
- With (a) quadratic polynomial: the boundary is well-estimated but may have some noise.
- With (b) BS put prices: the boundary is very clean because the basis functions capture the true shape extremely well.

**Comment:** The BS put price basis produces a much smoother and more accurate exercise boundary than quadratic polynomials. This is because the BS put already has the correct functional form; the regression only needs to calibrate two coefficients.

---

### Question 2.4: Bermudan-Asian Call Option

**Restatement:** Price a Bermudan-Asian call with $K = 100$, $T = 1$, monthly exercises, $r = q = 0$, $\sigma = 0.2$. Payoff at $t_n$: $\max(0, A_{t_n} - K)$ where $A_{t_n} = \frac{1}{n}\sum_{i=1}^n S_{t_i}$.

**(a)** Use basis functions: constant 1.0 and BS call with spot $Z_{t_n} = \frac{nA_{t_n} + (12-n)S_{t_n}}{12}$, strike $K$, maturity $T - t_n$, vol $\bar\sigma = 0.1$.

**Explanation of $Z_{t_n}$:** The final average $A_T = \frac{1}{12}\sum_{i=1}^{12} S_{t_i}$. At time $t_n$, we already know $S_{t_1}, \ldots, S_{t_n}$ but not $S_{t_{n+1}}, \ldots, S_{t_{12}}$. The quantity $Z_{t_n}$ approximates $A_T$ by:
- Using the known values $S_{t_1}, \ldots, S_{t_n}$ (contributing $nA_{t_n}$)
- Replacing the unknown future values $S_{t_{n+1}}, \ldots, S_{t_{12}}$ with the current spot $S_{t_n}$ (contributing $(12-n)S_{t_n}$)
- Dividing by 12

This gives a proxy for the final average that is observable at $t_n$. A BS call with this proxy spot captures the essential option-like shape of the continuation value.

Using $A_{t_n}$ alone would miss the sensitivity to $S_{t_n}$ (future paths depend on current spot). Using $S_{t_n}$ alone would miss the already-accumulated average. $Z_{t_n}$ combines both pieces of information.

**(b)** Neural network approach: Use a feedforward NN with inputs $(S_{t_n}, A_{t_n})$, 3 hidden layers of 20 neurons with ReLU activation. Train on 50,000 paths with batch size 128. Use validation loss for early stopping.

**Why NN for part (b):** The Bermudan-Asian option depends on *two* state variables $(S_{t_n}, A_{t_n})$ --- this is a path-dependent option. Polynomial regression in 2D becomes cumbersome. Neural networks handle multi-dimensional regression naturally via the universal approximation theorem.

---

# PART 3 --- "WHAT YOU MUST KNOW" SECTION

## A) Pure Theory to Memorize

1. **Snell envelope definition and recursive construction** --- This WILL be on any exam.
2. **Bermudan option value = Snell envelope of discounted payoff** --- The conceptual link.
3. **Optimal stopping time:** $\tau^*_k = \inf\{n \ge k : Z_n = U_n\}$.
4. **Doob decomposition:** $U_n = M_n - A_n$ (martingale + predictable decreasing).
5. **Primal-dual identity:** $\sup_\tau \mathbb{E}[D_{t,\tau}F_\tau|\mathcal{F}_t] = \inf_{M \in \mathcal{M}_{t,0}} \mathbb{E}[\sup_s (D_{t,s}F_s - M_s)|\mathcal{F}_t]$.
6. **Variational inequality** for American options in the Markovian case.
7. **Feynman-Kac theorem** (stochastic representation of PDE solutions).
8. **Super-replication price theorem:** buyer's/seller's prices as inf/sup over ELMM.
9. **Complete market $\iff$ unique ELMM.**
10. **Jensen's inequality** and its role in creating bias in MC American option pricing.

## B) Proof Techniques to Master

1. **Proving the variational inequality characterizes the American option price:**
   - Show $\mathcal{J}u \le 0$ implies discounted price is supermartingale $\Rightarrow$ upper bound.
   - Show the optimal stopping time $\tau^*$ achieves equality $\Rightarrow$ lower bound.

2. **Proving the Snell envelope recursion:**
   - Forward direction: from definition of $U_k$ as supremum, derive the recursive formula.
   - Backward direction: from the recursion, show $U_k$ is the smallest dominating supermartingale.

3. **Proving primal-dual equality:**
   - Weak duality: use optional sampling $\mathbb{E}[M_\tau|\mathcal{F}_t] = M_t = 0$ to show primal $\le$ dual.
   - Strong duality: construct $M^*$ from the Doob-Meyer decomposition of the Snell envelope.

4. **Proving the sufficient condition for no-arbitrage:**
   - Show discounted admissible portfolio wealth is a lower-bounded local martingale $\Rightarrow$ supermartingale $\Rightarrow$ $\mathbb{E}[\tilde\pi_T] \le 0$.

## C) Standard Patterns in Nonlinear Options Problems

1. **"Estimate continuation value, then compare with exercise value"** --- The universal pattern for Bermudan pricing.

2. **"Two-step procedure for lower bound":**
   - Step 1: Determine exercise policy (LS or TVR).
   - Step 2: Evaluate on fresh independent paths.

3. **"Dual upper bound via martingale construction":**
   - Approximate the Snell envelope.
   - Extract the martingale part via: $M_{t_{i+1}} = M_{t_i} + V_{t_{i+1}} - \mathbb{E}[V_{t_{i+1}}|\mathcal{F}_{t_i}]$.
   - Compute $\mathbb{E}[\max_i (D_{0,t_i}F_{t_i} - M_{t_i})]$.

4. **"Duality gap = quality diagnostic":** A large gap between lower and upper bounds means the exercise policy is poor (bad regression).

5. **"In-the-money restriction":** Always restrict regression to ITM paths in LS --- it costs nothing and improves accuracy.

6. **"Choose basis functions that mimic the payoff":** BS prices as basis functions outperform polynomials because they already have the right shape.

## Typical Exam Tricks

- **Asking you to prove the Snell envelope recursion** (both directions).
- **Asking you to explain why LS gives a lower bound and TVR has undetermined bias.**
- **Asking you to derive the variational inequality** from first principles.
- **Asking you to identify continuation and exercise regions** from the variational inequality.
- **Asking you to compute/explain the Doob-Meyer decomposition** in a specific example.
- **Giving you a nonlinear PDE and asking you to identify the optimization/nonlinearity source.**
- **Path-dependent option pricing:** adding an auxiliary state variable to make the problem Markovian.
- **Asking about the relationship between market completeness and the uniqueness of ELMM.**

## What is Almost Always Tested

1. Snell envelope and its recursive construction.
2. Longstaff-Schwartz algorithm (implementation and bias analysis).
3. Lower bound / upper bound pricing.
4. Conditional expectation estimation techniques (regression, kernels).
5. Variational inequality for American options.

## What to Write to Get Full Points

- **Always state the theorem/result you are using before applying it.**
- **Always verify assumptions** (integrability, adaptedness, supermartingale property, etc.).
- **Use precise notation** ($\mathbb{E}^Q[\cdot|\mathcal{F}_t]$, $D_{t,T}$, $\mathcal{T}_{t,T}$).
- **Explain the intuition** --- why does this tool work here?
- **For bias analysis:** explicitly invoke Jensen's inequality and the convexity of $\max$.
- **For completeness/incompleteness questions:** explicitly identify the number of assets vs. Brownian motions and the invertibility of $\sigma$.

---

# PART 4 --- TOOL RECALL TABLE

| Exercise/Question | Tool Used | Why It Was Needed | Underlying Theorem | Problem Type |
|---|---|---|---|---|
| Q1.1 (Parametric regression) | Polynomial regression, Piecewise-linear regression | Estimate $\mathbb{E}[Y\|X]$ from data by projecting onto a finite-dimensional function space | Conditional expectation as $L^2$ best predictor; Bias-variance tradeoff | Conditional expectation estimation |
| Q1.2 (Nonparametric regression) | Nadaraya-Watson kernel regression, Local linear regression | Estimate $\mathbb{E}[Y\|X]$ without assuming a parametric form | Kernel density estimation; $\mathbb{E}[Y\|X=x] = \int y f_{Y|X}(y|x)\,dy$; Silverman's rule | Conditional expectation estimation |
| Q2.1(a) (BS put basis) | Parametric regression with BS prices as basis functions | Estimate continuation value in Snell envelope recursion | Snell envelope recursive construction; LS/TVR algorithms | Optimal stopping (Bermudan put) |
| Q2.1(b) (Piecewise linear) | Piecewise linear regression for continuation value | Same as above, alternative regression method | Same as above | Optimal stopping (Bermudan put) |
| Q2.1(c) (Kernel regression) | Nadaraya-Watson with Gaussian kernel | Same as above, nonparametric alternative | Same as above | Optimal stopping (Bermudan put) |
| Q2.2 (Lower bound) | Two-step Monte Carlo: policy determination + independent evaluation | Obtain unbiased lower bound on American option price | Suboptimality principle ($\mathbb{E}[D_{0,\tau}F_\tau] \le u_0$ for any $\tau$); Jensen's inequality (explains why single-step has bias) | Optimal stopping (lower bound pricing) |
| Q2.3 (ITM regression) | Longstaff-Schwartz with in-the-money restriction | Improve regression accuracy by focusing on the exercise-relevant region | Same Snell envelope recursion; OTM always continues ($F = 0 < C$) | Optimal stopping (improved LS) |
| Q2.4(a) (Bermudan-Asian) | LS with BS call proxy basis; Path-dependent option pricing | Price a Bermudan-Asian call; need proxy spot $Z_{t_n}$ to capture path-dependency | Snell envelope; Markovianization via auxiliary variable $A_{t_n}$ | Optimal stopping (path-dependent) |
| Q2.4(b) (Neural network) | Feedforward neural network for continuation value estimation | Handle 2D state space $(S_{t_n}, A_{t_n})$ where polynomial regression is cumbersome | Universal Approximation Theorem; Snell envelope recursion | Optimal stopping (neural network / ML) |

---

## Final Remarks

The central message of this course is:

> **Nonlinear option pricing = Optimization over strategies/measures/controls.**

The linearity of Black-Scholes comes from the uniqueness of the replicating portfolio in a complete market. As soon as you introduce optimization --- choosing when to exercise (American), choosing worst-case volatility (uncertain volatility), choosing between borrowing and lending rates --- the pricing problem becomes nonlinear.

The mathematical tools (Snell envelope, BSDEs, viscosity solutions, convex duality) are all different lenses for studying the same nonlinear PDE or nonlinear expectation operator. The computational tools (Monte Carlo + regression/ML) are the practical machinery for solving these problems in the high-dimensional settings that arise in practice.

Master the Snell envelope recursion and the LS algorithm. Everything else builds on these foundations.
