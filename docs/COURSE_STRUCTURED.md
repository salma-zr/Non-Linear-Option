## Nonlinear Option Pricing — Structured Course Notes (from `NonlinearOptionPricing_LectureNotes_SorbonneUniversite_2026.pdf`)

This document reorganizes the lecture into a clean structure and extracts the **key definitions, theorems, tools, and deep ideas** used in nonlinear/early-exercise pricing.

### Scope and philosophy

- **Linear pricing** (complete markets, single EMM, Feynman–Kac) is the baseline.
- **Nonlinear pricing** appears when *one* of the following breaks linearity:
  - **early exercise** (optimal stopping → obstacle/variational inequality),
  - **market incompleteness / constraints / frictions** (super-replication, convex risk measures, multiple martingale measures → nonlinear expectation),
  - **stochastic control** (optimization over controls → HJB / dynamic programming),
  - **nonlinear conditional expectations** (BSDEs, \(g\)-expectations).
- **Numerical focus** is Monte Carlo / regression / ML due to curse of dimensionality.

---

## Part A — Core mathematical objects (what they are, why they matter)

### A1) Optimal stopping and American/Bermudan options

#### Problem template

Given a filtered probability space and discounted payoff process \(Z_t := D_{0,t} F_t\), the American value is
\[
V_0=\sup_{\tau\in\mathcal T_{0,T}}\mathbb E[Z_\tau],
\quad
V_t=\operatorname*{ess\,sup}_{\tau\in\mathcal T_{t,T}}\mathbb E[Z_\tau\mid\mathcal F_t].
\]

#### Tool 1 — Snell envelope (discrete time in the lecture; continuous time conceptually)

- **Definition (discrete time)**: for \((Z_k)_{1\le k\le N}\),
\[
U_k:=\sup_{\tau\in\mathcal T_{k,N}}\mathbb E[Z_\tau\mid\mathcal F_k].
\]
- **Recursive construction**:
\[
U_N=Z_N,\qquad
U_k=\max\{Z_k,\ \mathbb E[U_{k+1}\mid\mathcal F_k]\}.
\]
- **Deep idea**: the American/Bermudan price is the **smallest supermartingale dominating** the payoff.
- **Used for**:
  - defining the value process rigorously,
  - proving optimality of “first hitting time of the exercise region,”
  - turning early-exercise into a backward recursion requiring **conditional expectations**.

#### Tool 2 — Optional sampling / supermartingale property

If \(D_{0,t}u(t,X_t)\) is a supermartingale and \(u\ge g\), then \(\mathbb E[D_{0,\tau}g(X_\tau)\mid\mathcal F_t]\le D_{0,t}u(t,X_t)\) for all stopping times \(\tau\).

- **Used for**: proving that a candidate solution of an obstacle PDE/variational inequality is an American price (verification argument).

---

### A2) Markovian PDE viewpoint: variational inequality / obstacle problem

Assume \(X_t\) Markov with generator \(L\), rate \(r\), payoff \(g\).

#### Tool 3 — Variational inequality (American in Markov setting)

The value \(u(t,x)\) solves (formally; rigor needs viscosity solutions):
\[
\max\big(\partial_t u + Lu - r u,\ g-u\big)=0,\qquad u(T,x)=g(x).
\]

- **Continuation region**: \(u>g\) ⇒ \(\partial_t u+Lu-ru=0\).
- **Exercise region**: \(u=g\).
- **Complementarity**: \((\partial_t u+Lu-ru)(g-u)=0\) (where smooth).

#### What is conceptually important here?

- The inequality is **not just a PDE trick**: it encodes the *economic fact* that the holder chooses between “exercise now” and “continue.”
- The nonlinear feature is the **max** operator (an optimizer) → that is why the PDE becomes nonlinear even in a linear diffusion model.

#### Tool 4 — Viscosity solutions (why they appear)

Free-boundary/obstacle problems typically lack smoothness at the exercise boundary. Viscosity solutions provide:
- a notion of solution without \(C^{1,2}\) regularity,
- comparison/uniqueness principles,
- convergence guarantees for monotone schemes (Barles–Souganidis paradigm).

(The lecture notes present the smooth “formal proof” first; for exams, you should know *where* smoothness is used and how viscosity replaces it.)

---

### A3) Doob–Meyer / Doob decomposition: separating “martingale” vs “exercise premium”

#### Tool 5 — Doob–Meyer decomposition (continuous time) / Doob decomposition (discrete time)

For the Snell envelope \(U\) (smallest supermartingale dominating payoff), there is a decomposition
\[
U_t = M_t - A_t,
\]
where \(M\) is a martingale and \(A\) is predictable increasing (and \(A_0=0\)).

- **Intuition**:
  - \(M\) is the “fair game” part (hedgeable / no-arbitrage martingale piece),
  - \(A\) is the “compensator” that enforces the supermartingale constraint and carries the **early-exercise premium**.
- **Used for**:
  - duality upper bounds (Rogers / Haugh–Kogan),
  - constructing near-optimal martingales from approximate value processes (Monte Carlo dual methods).

---

### A4) Primal–dual duality for American options (Rogers; Haugh–Kogan)

#### Tool 6 — Rogers / Haugh–Kogan dual representation

For discounted payoff \(Z_t=D_{0,t}F_t\),
\[
\sup_{\tau\in\mathcal T_{t,T}}\mathbb E[Z_\tau\mid\mathcal F_t]
=
\inf_{M\in\mathcal M_{t,0}}\mathbb E\Big[\sup_{s\in[t,T]}(Z_s-M_s)\,\Big|\,\mathcal F_t\Big],
\]
where \(\mathcal M_{t,0}\) is the set of right-continuous martingales with \(M_t=0\).

- **Why it applies**:
  - optional sampling gives, for any \(\tau\), \(\mathbb E[M_\tau\mid\mathcal F_t]=0\),
  - taking sup over \(\tau\) and then inf over \(M\) yields weak duality,
  - equality holds by choosing \(M^\*\) as the **martingale part** of the Snell envelope (Doob–Meyer).
- **Assumptions to remember**:
  - integrability so expectations exist,
  - right-continuity / usual conditions for Doob–Meyer.

#### Deep exam idea

- **Lower bound**: any admissible stopping time / exercise strategy gives a **lower bound**.
- **Upper bound**: any martingale \(M\) with \(M_0=0\) gives an **upper bound** via \(\mathbb E[\sup(Z-M)]\).
- **Duality gap** diagnoses numerical quality: a large gap usually means poor regression/exercise rules.

---

### A5) Super-replication and nonlinear expectations (incomplete markets / frictions)

#### Tool 7 — Equivalent local martingale measures (ELMM)

An ELMM \(Q\) is a probability measure equivalent to historical \(P\) under which discounted traded assets are local martingales.

#### Tool 8 — Super-replication price theorem (El Karoui–Quenez; Kramkov)

In incomplete markets (or with constraints), perfect replication fails and the **buyer/seller super-replication prices** satisfy:
\[
B_t(F_T)=\inf_{Q\in\mathrm{ELMM}}\mathbb E^Q[D_{t,T}F_T\mid\mathcal F_t],
\qquad
S_t(F_T)=\sup_{Q\in\mathrm{ELMM}}\mathbb E^Q[D_{t,T}F_T\mid\mathcal F_t].
\]

- **Intuition**:
  - you must be robust across all risk-neutral measures compatible with no-arbitrage,
  - the “price” becomes a **nonlinear expectation** (inf/sup over measures).
- **Deep idea**: nonlinearity comes from **model uncertainty / incompleteness**, not from payoff convexity.

---

### A6) BSDEs, stochastic control, and nonlinear PDEs (high-level map)

The lecture syllabus lists:
- **stochastic control** → dynamic programming → HJB nonlinear PDE,
- **BSDEs** (often with comparison principles) → nonlinear expectations / semilinear PDE,
- reflected BSDEs ↔ obstacle problems (American options),
- McKean SDEs / particle methods and branching diffusions for high-dimensional PDE/BSDE approximations.

For exam readiness: you should be able to explain *which source of nonlinearity* you are in:
- max/sup over stopping times (optimal stopping),
- sup/inf over controls (control),
- inf/sup over measures (duality / super-replication),
- nonlinear driver in a BSDE (\(g\)-expectation).

---

## Part B — Numerical “core workflow” (what gets computed and why)

### B1) Bermudan approximation + backward induction

American → Bermudan on grid \(t_1<\dots<t_N\). Backward recursion:
\[
V_{t_N}=F_{t_N},\qquad
V_{t_i}=\max\Big(F_{t_i},\ \mathbb E[D_{t_i,t_{i+1}}V_{t_{i+1}}\mid\mathcal F_{t_i}]\Big).
\]

**Bottleneck**: estimating conditional expectations in high dimension.

### B2) Regression Monte Carlo (Longstaff–Schwartz; Tsitsiklis–Van Roy)

- **Common structure**: simulate paths; regress discounted continuation values on features/basis functions; exercise if immediate payoff exceeds estimated continuation.
- **Deep numerical idea**: the whole method is a *projection of conditional expectation* onto a chosen function class (polynomials, piecewise linear, kernels, neural nets).

### B3) Dual Monte Carlo (upper bounds)

- Build an approximate value process \(\hat U\).
- Extract a martingale part \(\hat M\) via (discrete) Doob decomposition.
- Compute \(\mathbb E[\sup(Z-\hat M)]\) as an **upper bound**.

---

## Part C — What to prioritize (deep ideas vs technical details)

### C1) The “big 4” structural templates

- **Optimal stopping**: value = Snell envelope = obstacle PDE/VI.
- **Duality**: primal over stopping times ↔ dual over martingales; Doob–Meyer connects them.
- **Robust/incomplete markets**: price = nonlinear expectation (inf/sup over ELMM).
- **Control/BSDE**: nonlinearity from optimization or nonlinear drivers; PDE link via HJB/semilinear PDE.

### C2) What is usually technical detail

- exact smoothness assumptions needed to justify Itô + optional sampling in the VI proof,
- discretization details of finite-difference schemes (unless explicitly asked),
- ML tuning/hyperparameters (conceptual role: conditional expectation approximator).

---

## Part D — Quick tool-to-problem index

- **American/Bermudan early exercise**: Snell envelope, optional sampling, VI/obstacle, regression MC, dual martingale bounds.
- **Upper/lower bounds diagnostics**: Rogers/Haugh–Kogan duality + Doob(-Meyer) decomposition.
- **Incomplete markets / frictions**: ELMM set + super-replication theorem (inf/sup over measures).
- **Control / nonlinear PDE**: dynamic programming, HJB, viscosity solutions, comparison.
