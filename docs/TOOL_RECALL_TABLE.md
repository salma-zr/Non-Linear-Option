## Tool recall table (per HW1 question)

This table maps each question to the structural tool(s), the underlying theorem, and the problem type.

| Question | Type | Tool(s) used | Why needed | Underlying theorem/result |
|---|---|---|---|---|
| 1.1 | Nonlinear expectation / regression (numerical primitive) | Conditional expectation as \(L^2\)-projection; OLS least squares | Regression is approximating \(\mathbb E[Y\mid X]\); over/underfitting = bias/variance | Orthogonal projection property of conditional expectation; OLS normal equations |
| 1.2 | Nonparametric estimation | Kernel regression (Nadaraya–Watson / local linear); bandwidth selection | Nonparametric approximation of \(\mathbb E[Y\mid X=x]\); shows smoothing vs noise-fitting | Bias–variance scaling for kernel estimators (heuristics); consistency under i.i.d. sampling |
| 2.1(a) | Bermudan/American pricing (optimal stopping, MC) | Snell recursion + regression on engineered basis (European BS put) | Approximate continuation value efficiently with informative feature | Discrete-time Snell envelope recursion \(V=\max(F,\mathbb E[\cdot\mid\mathcal F])\) |
| 2.1(b) | Same | Piecewise-linear regression (spline/hat basis) | Flexible but controlled approximation of continuation value | Same Snell recursion; regression as projection |
| 2.1(c) | Same | Kernel regression for continuation value | Nonparametric conditional expectation estimator | Same Snell recursion; kernel smoothing consistency (1D) |
| 2.2 | Lower bound justification | “Any stopping time yields lower bound”; independent policy evaluation | Proves correctness of lower bound and separates training vs evaluation bias | Definition \(V_0=\sup_{\tau}\mathbb E[Z_\tau]\); LLN for MC estimator |
| 2.3 | Numerical stability of LSM | ITM regression restriction; classification of exercise vs continuation | Reduces noise/variance where payoff is 0; improves boundary stability | Same Snell recursion; dominance argument “OTM ⇒ never exercise” |
| 2.4(a) | Path-dependent Bermudan (Markovization) | State reduction via proxy \(Z_{t_n}\); LSM regression; out-of-sample lower bound | Makes Asian payoff more “Markov-like” for regression; produces usable policy | Dynamic programming on grid; admissible policy ⇒ lower bound |
| 2.4(b) | High-dimensional conditional expectation approximation | Neural network regression for continuation value | Learns nonlinear basis functions for \(\mathbb E[\cdot\mid S,A]\) | Universal approximation theorem (intuition); ERM/MSE regression principle |

### Notes for grading-oriented use

- If a question asks “why bound?”, write **one line**: “because \(\hat\tau\) is an admissible stopping time, so \(\mathbb E[Z_{\hat\tau}]\le \sup_\tau \mathbb E[Z_\tau]\).”
- If a question asks “which theorem?”, cite **Snell envelope** for recursion; cite **optional sampling** / “sup over stopping times” for lower bounds; cite **Doob decomposition** if constructing dual martingale bounds.
