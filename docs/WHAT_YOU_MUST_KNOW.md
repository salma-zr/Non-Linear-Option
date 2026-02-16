## “What you must know” — Nonlinear Option Pricing (exam sheet)

This is a compact checklist to maximize points under time pressure.

---

## A) Pure theory to memorize (definitions + statements)

### A1) Optimal stopping / American options

- **American price (discounted payoff \(Z_t\))**:
\[
V_t=\operatorname*{ess\,sup}_{\tau\in\mathcal T_{t,T}}\mathbb E[Z_\tau\mid\mathcal F_t].
\]

- **Snell envelope (discrete time)**:
  - definition \(U_k=\sup_{\tau\in\mathcal T_{k,N}}\mathbb E[Z_\tau\mid\mathcal F_k]\),
  - recursion \(U_N=Z_N,\ U_k=\max(Z_k,\mathbb E[U_{k+1}\mid\mathcal F_k])\),
  - smallest supermartingale dominating \(Z\),
  - optimal \(\tau_k^\*=\inf\{n\ge k:\ Z_n=U_n\}\).

- **Variational inequality (Markov case)**:
\[
\max(\partial_t u+Lu-ru,\ g-u)=0,\qquad u(T,\cdot)=g.
\]

### A2) Doob(-Meyer) decomposition

- **Discrete Doob decomposition**: any adapted integrable process can be written as martingale + predictable part.
- **Doob–Meyer (continuous)**: a (right-continuous) supermartingale decomposes uniquely as
\[
U_t=M_t-A_t,\quad A_t \text{ increasing predictable}.
\]

### A3) Primal–dual duality (Rogers; Haugh–Kogan)

\[
\sup_{\tau}\mathbb E[Z_\tau\mid\mathcal F_t]
=
\inf_{M\in\mathcal M_{t,0}}\mathbb E\Big[\sup_{s\in[t,T]}(Z_s-M_s)\mid\mathcal F_t\Big].
\]
Optimal \(M^\*\) = martingale part of Snell envelope.

### A4) Super-replication in incomplete markets (El Karoui–Quenez; Kramkov)

\[
B_t(F_T)=\inf_{Q\in\mathrm{ELMM}}\mathbb E^Q[D_{t,T}F_T\mid\mathcal F_t],
\quad
S_t(F_T)=\sup_{Q\in\mathrm{ELMM}}\mathbb E^Q[D_{t,T}F_T\mid\mathcal F_t].
\]

### A5) Dynamic programming / control / viscosity solutions (high-level)

- Control ⇒ HJB nonlinear PDE; value function characterized by DPP + viscosity solution.
- Viscosity comparison principle ⇒ uniqueness and stability of numerical schemes.

---

## B) Proof techniques to master (how to score on “show that…”)

### B1) Verification by supermartingale + optional sampling

To prove a candidate \(u\) is an American price:
- show \(u\ge g\),
- show \(D_{0,t}u(t,X_t)\) is a supermartingale (from \(\partial_t u+Lu-ru\le 0\)),
- apply optional sampling for any \(\tau\): \(\mathbb E[D_{0,\tau}g(X_\tau)\mid\mathcal F_t]\le D_{0,t}u(t,X_t)\),
- build \(\tau^\*=\inf\{u=g\}\) and show equality on \([t,\tau^\*]\) by “\(Ju=0\)” in continuation.

### B2) Discrete Snell recursion derivation

You must be able to derive
\[
V_{t_i}=\max\big(F_{t_i},\ \mathbb E[D_{t_i,t_{i+1}}V_{t_{i+1}}\mid\mathcal F_{t_i}]\big)
\]
from the definition as a supremum over discrete stopping times.

### B3) Duality inequality chain (for American)

For any \(\tau\) and martingale \(M\):
- optional sampling ⇒ \(\mathbb E[M_\tau\mid\mathcal F_t]=0\),
- hence \(\mathbb E[Z_\tau\mid\mathcal F_t]\le \mathbb E[\sup_s(Z_s-M_s)\mid\mathcal F_t]\),
- then take \(\sup_\tau\) then \(\inf_M\).

### B4) “Lower bound = any admissible policy”

Whenever you can define a stopping time / control / hedge that is admissible, its value is ≤ optimum.
This is the clean way to justify “lower bound” in Monte Carlo.

---

## C) Standard patterns in nonlinear options problems (what to write)

### C1) Early exercise pricing pattern

- Define payoff process \(Z\).
- Write value as \(\sup_\tau \mathbb E[Z_\tau]\).
- Invoke Snell envelope recursion (discrete) or obstacle/VI (Markov).
- Identify continuation vs exercise region.

### C2) Regression Monte Carlo (LSM/TVR) pattern

- Approximate American by Bermudan.
- Backward induction; at each step estimate conditional expectation by regression:
  - choose basis/features,
  - least squares fit,
  - exercise if immediate payoff ≥ estimated continuation.
- Report bias discussion:
  - out-of-sample evaluation of policy ⇒ lower bound,
  - dual martingale method ⇒ upper bound,
  - gap ⇒ quality diagnostic.

### C3) Robust / incomplete market pattern

- State ELMM set.
- Write buyer/seller super-replication prices as inf/sup of \(Q\)-expectations.
- Explain: nonlinearity = optimization over measures.

---

## Typical exam tricks / common traps

- **Trap**: calling an in-sample LSM recursion value a “lower bound” (needs out-of-sample evaluation).
- **Trap**: forgetting discounting in Snell recursion.
- **Trap**: mixing measures (simulate under \(\mathbb P\) but discount as if under \(\mathbb Q\)).
- **Trick**: show supermartingale domination ⇒ lower/upper bounds immediately.
- **Trick**: when asked “why nonlinear?” answer: “because of max/sup/inf (optimization) or nonlinear expectation/BSDE driver.”

---

## What is almost always tested

- Snell envelope recursion + optimal stopping time characterization.
- Doob(-Meyer) decomposition and its use in dual bounds.
- Why an independently evaluated policy gives a lower bound.
- Super-replication dual representation over ELMM (concept + correct formula).
