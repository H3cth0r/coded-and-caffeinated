# Quant Finance — A Project-Based Learning Course

A hands-on course where every topic is learned by building small, focused projects
that accumulate into a personal quant library.

---

## Philosophy

Three rules govern this course:

1. **Measure before you optimize.** You can't build a max-Sharpe portfolio if you
   don't understand what Sharpe, beta, and alpha actually measure. The first
   projects build the measurement toolkit; optimization comes later.
2. **Implement from scratch → use the library → compare.** For every major
   concept we write it ourselves first (learning), then use the professional
   library version (validation), then compare results (understanding). Seeing
   your hand-rolled optimizer match CVXPY is how you know you actually get it.
3. **Everything becomes part of a library.** This course doesn't produce 14
   throwaway projects — it produces `qfin`, a personal quant library you wrote
   and understand line by line.

### The per-project workflow

Every project follows the same loop:

1. **Prototype** in a notebook — messy is fine, the goal is understanding.
2. **Promote** the clean, reusable parts into `qfin/` as a module with a
   docstring and a small sanity test.
3. **Reuse** — the next project imports from `qfin/` instead of copy-pasting.
4. **Tag** — commit and tag the repo (`v0.1-metrics`, `v0.2-factors`, ...) so the
   library's growth is visible in git history.

### Integration hooks

Projects are sequenced so that each one *feeds* later ones. Examples:

- The **Kalman filter** (project 04) upgrades project 01's static beta into a
  time-varying beta.
- **DCF outputs** (project 03) become the *views* for Black-Litterman
  (project 09) and a *value factor* for factor-based allocation.
- **Monte Carlo** (project 10) powers a Monte-Carlo VaR in project 07.
- **K-means clustering** (project 12) feeds the hierarchical tree in HRP
  (project 09).
- The **capstone** (project 14) stitches all of it into one complete strategy.

---

## Repository structure

```
quant-finance/
├── README.md                 ← this course plan
├── pyproject.toml            ← package metadata (pip install -e .)
├── qfin/                     ← THE LIBRARY (the thing that accumulates)
│   ├── data/                 ← providers: massive.py, yfinance.py, cache.py
│   ├── metrics/              ← sharpe, sortino, calmar, drawdown, beta, alpha...
│   ├── factors/              ← CAPM, Fama-French regressions, factor exposure
│   ├── valuation/            ← DCF, multiples, comps
│   ├── optimization/         ← MVO, solvers, constraints, risk parity, HRP, BL
│   ├── risk/                 ← covariance, shrinkage, VaR/CVaR, GARCH, stress
│   ├── simulation/           ← MC, QMC, LHS, variance reduction, integration
│   ├── timeseries/           ← stationarity, ARIMA, Kalman, regime switching
│   ├── backtest/             ← backtester, attribution, walk-forward
│   └── ml/                   ← clustering, PCA, feature selection, RL
├── tests/                    ← a small sanity test per promoted module
└── projects/                 ← the practice ground (thin, one dir per project)
    ├── 00-data-toolkit/
    │   ├── README.md         ← anchor question, topics, notes
    │   ├── notebooks/        ← exploration (not part of the library)
    │   └── scripts/          ← small runnable entry points
    ├── 01-risk-return-metrics/
    └── ...
```

Notebooks stay in the projects; only promoted `.py` modules go into `qfin/`.

---

## Documentation

Per-project implementation plans and reference docs live in [`docs/`](docs/index.md):

- [Project 00 — Data Toolkit: Implementation Plan](docs/project-00-data-toolkit.md)

---

## Environment & setup

- **Python environment**: one venv for the whole repo (`python -m venv .venv`).
  Use **uv** to create it and for fast installs (`uv venv`, `uv pip install ...`);
  use plain **pip** for everyday installs afterward.
- **Editable install**: `pip install -e .` from the repo root — edit `qfin/`,
  and every project sees the change immediately.
- **Data**: **Massive (formerly Polygon.io)** as the primary provider, with
  **yfinance** as a free fallback. All data is cached to local parquet/CSV so
  each series is downloaded exactly once.
- **Core libraries**: NumPy, SciPy, pandas, matplotlib, statsmodels,
  scikit-learn, CVXPY, PyPortfolioOpt, vectorbt, arch (GARCH), yfinance,
  massive/polygon client.
- Optional/later: Pyomo, Gurobi/CPLEX/MOSEK (commercial solvers — ECOS/SCS are
  free and sufficient for everything here), CuPy for GPU experiments.

## Data provider abstraction (built in project 00)

```python
class PriceProvider(Protocol):
    def get_prices(self, tickers, start, end) -> pd.DataFrame: ...
```

Two implementations: `MassiveProvider` and `YFinanceProvider`, plus a
`CachedProvider` wrapper that persists every response to disk. Every later
project is data-source-agnostic.

---

## The course

### ☐ Project 00 — Data Toolkit
**Anchor question:** *How do I get clean, cached price data for any ticker?*

- Build the provider interface, `MassiveProvider`, `YFinanceProvider`,
  caching layer, and helpers for returns (simple/log), alignment, and
  resampling.
- **Topics:** data handling, real-time vs historical data, benchmark selection,
  caching.
- **Promote:** the entire `qfin/data/` package.
- **Deliverable:** a script that fetches and caches a small universe
  (e.g., SPY benchmark + 20 stocks) in one command.

### ☐ Project 01 — Risk & Return Metrics
**Anchor question:** *Which of these portfolios actually performed better?*

- Returns, volatility, drawdowns, Sharpe / Sortino / Calmar, VaR (first pass),
  **beta & alpha** via OLS regression against SPY, rolling beta/alpha, t-tests
  on alpha ("is this alpha real or luck?").
- **Topics:** Sharpe ratio & risk-adjusted returns, risk-return tradeoff,
  backtesting statistics, hypothesis testing, beta & alpha, benchmarking.
- **Promote:** `qfin/metrics/` (sharpe, sortino, calmar, drawdown, beta, alpha).
- **Deliverable:** a report comparing 3–4 portfolios (equal weight, cap weight,
  a single stock, SPY) with a full metrics table.
- **Feeds into:** everything. These metrics are used in every later project.

### ☐ Project 02 — CAPM, Beta/Alpha & Factor Models
**Anchor question:** *Is my alpha real, or just exposure to known factors?*

- CAPM in depth, Fama-French 3-factor and 5-factor regressions, APT
  intuition, factor exposure decomposition, multi-factor alpha, statistical
  significance of alpha after controlling for factors.
- **Topics:** CAPM, factor models (Fama-French, APT), risk factor
  decomposition, beta & alpha (deep dive).
- **Promote:** `qfin/factors/` (CAPM regression, FF factor regressions,
  rolling factor exposure).
- **Deliverable:** regression report for a few stocks/ETFs showing how their
  alpha shrinks once size/value/momentum factors are controlled for.
- **Feeds into:** factor-based allocation (09), risk factor decomposition (09),
  risk-model covariance via factor models (07).

### ☐ Project 03 — DCF & Valuation Multiples
**Anchor question:** *What is this company actually worth?*

- Build a DCF engine: revenue drivers → FCF projection, WACC (CAPM links back
  to project 02 for the cost of equity!), terminal value (Gordon growth vs.
  exit multiple), sensitivity tables (WACC × growth). Then relative valuation:
  P/E, EV/EBITDA, P/B, P/S, PEG, comps tables and scatter plots.
- **Topics:** DCF modeling, valuation multiples, WACC, terminal value,
  sensitivity analysis, Taylor-series intuition in sensitivity approximations.
- **Promote:** `qfin/valuation/` (dcf.py, multiples.py, comps.py).
- **Deliverable:** a DCF + comps valuation of 3 real companies, with a
  sensitivity heatmap and a "fair value vs market price" table.
- **Feeds into:** Black-Litterman *views* (09), a *value factor* for factor
  allocation (09), fundamentals for the capstone.

### ☐ Project 04 — Time Series, Kalman Filters & Regimes
**Anchor question:** *Are returns stationary — and what regime am I in right now?*

- Stationarity (ADF/KPSS), AR/ARIMA with statsmodels, the Kalman filter
  implemented from scratch then compared — use it to estimate a *time-varying
  beta* (upgrading project 01's static OLS beta), regime-switching with a
  simple HMM on bull/bear/sideways markets.
- **Topics:** time series analysis & stationarity, AR/ARIMA, Kalman filtering,
  regime switching models, dynamic beta/alpha.
- **Promote:** `qfin/timeseries/` (stationarity tests, arima wrappers,
  kalman.py, regime detection).
- **Deliverable:** a chart of SPY with detected regimes and a stock's rolling
  Kalman beta vs. its static OLS beta.
- **Feeds into:** volatility forecasting context (07), dynamic asset
  allocation (09), regime-aware signals for the capstone.

### ☐ Project 05 — Mean-Variance & the Efficient Frontier
**Anchor question:** *What is the best mix of these 10 assets?*

- Markowitz theory, mean-variance optimization with SciPy SLSQP, minimum
  variance portfolio, maximum Sharpe portfolio, the full efficient frontier,
  comparison vs. equal weight and cap weight. Write the optimizer from scratch
  (including a naive grid search) to build intuition before trusting SLSQP.
- **Topics:** Modern Portfolio Theory, mean-variance optimization, efficient
  frontier, min-variance portfolio, max-Sharpe portfolio, SLSQP, convexity.
- **Promote:** `qfin/optimization/mvo.py`.
- **Deliverable:** efficient frontier plot with min-var, max-Sharpe, equal
  weight, and market-cap portfolios marked; weights tables for each.
- **Feeds into:** the constrained versions in 08; the baseline all other
  allocation methods are judged against.

### ☐ Project 06 — Solver & Optimization Lab
**Anchor question:** *Why do five solvers give five different answers?*

- Reimplement project 05's MVO in CVXPY; compare ECOS, SCS, and SciPy SLSQP
  (and a Gurobi/CPLEX free trial if desired). Demos of LP vs. QP vs. SOCP vs.
  SDP, gradient descent and SGD written from scratch on the same problem,
  Lagrangian intuition by solving constrained problems as penalized ones,
  interior-point vs. active-set behavior, convergence plots.
- **Topics:** convex optimization, QP, LP, SDP, SOCP, Lagrangian relaxation,
  gradient descent & SGD, interior point methods, SLSQP, solver selection
  (ECOS/SCS/CPLEX/Gurobi/MOSEK), algorithm complexity.
- **Promote:** `qfin/optimization/solvers.py`, `qfin/optimization/cvx.py`.
- **Deliverable:** a benchmark notebook: same problem, N solvers — runtime,
  accuracy, and robustness compared.
- **Feeds into:** every optimization project afterward uses the best tool for
  the job, not just the first one that worked.

### ☐ Project 07 — Risk Modeling
**Anchor question:** *How much can I lose on a bad day?*

- Covariance matrix estimation and its pitfalls (N > T instability), correlation
  matrices, **Ledoit-Wolf shrinkage** (implemented from scratch vs.
  sklearn), factor-model covariance (from project 02's factors), VaR and CVaR
  three ways (historical, parametric, Monte Carlo — MC comes from project 10),
  EWMA volatility, GARCH with `arch`, tail risk and extreme value intuition,
  stress testing (replay 2008, COVID, rate shocks).
- **Topics:** covariance estimation, correlation matrices, shrinkage
  estimators (Ledoit-Wolf), factor-model covariance, VaR, CVaR, GARCH/EWMA,
  tail risk, stress testing.
- **Promote:** `qfin/risk/` (covariance.py, shrinkage.py, var.py, garch.py,
  stress.py).
- **Deliverable:** risk report on the project 05 portfolio: VaR/CVaR under
  three methods, GARCH volatility forecast, and stress-test P&L.
- **Feeds into:** better covariance = better MVO (08); CVaR becomes an
  optimization objective in 08; stress tests feed the capstone's risk checks.

### ☐ Project 08 — Constraints & Real-World Frictions
**Anchor question:** *Why does my "optimal" portfolio lose money in practice?*

- Re-solve MVO with box constraints (position limits), the full-investment
  constraint, turnover constraints, sector constraints, short-sale bans, and
  minimum position sizes. Model transaction costs (linear and quadratic
  impact) and slippage, and see how the "optimal" frontier collapses once
  costs are real. Cardinality constraints (max K assets) via MILP or greedy
  heuristics.
- **Topics:** box constraints, linear constraints, turnover constraints,
  transaction costs modeling, slippage estimation, cardinality constraints,
  sector constraints, short-sale restrictions, minimum position sizes,
  CVXPY mixed-integer programs.
- **Promote:** `qfin/optimization/constraints.py`, `qfin/optimization/costs.py`.
- **Deliverable:** frontier plots with and without constraints/costs, plus a
  table showing how each constraint changes weights, turnover, and net return.
- **Feeds into:** the capstone's production optimizer is exactly this.

### ☐ Project 09 — Advanced Asset Allocation
**Anchor question:** *How do professionals allocate when they disagree with the market?*

- Risk parity (naive risk budgeting + convex formulation), Hierarchical Risk
  Parity, Black-Litterman — with **views sourced from your project 03 DCF
  outputs** — smart beta strategies, strategic vs. tactical vs. dynamic
  allocation, factor-based allocation using project 02's factor scores, equal
  weight vs. cap weight revisited rigorously.
- **Topics:** risk parity, HRP, Black-Litterman, smart beta, strategic/
  tactical/dynamic allocation, factor-based allocation.
- **Promote:** `qfin/optimization/risk_parity.py`, `hrp.py`,
  `black_litterman.py`, `smart_beta.py`.
- **Deliverable:** head-to-head comparison of all allocation schemes on the
  same universe and period, using project 01's metrics library for scoring.
- **Feeds into:** allocation strategies for the capstone; HRP becomes a
  baseline for ML clustering in 12.

### ☐ Project 10 — Monte Carlo & Numerical Methods
**Anchor question:** *How do I measure things that have no closed-form answer?*

- Monte Carlo simulation of portfolio paths (GBM, then bootstrap returns),
  variance reduction (antithetic variates, control variates), quasi-Monte
  Carlo (Sobol sequences) vs. plain MC convergence, Latin Hypercube Sampling,
  importance sampling for tail probabilities, numerical integration (SciPy
  quadrature vs. MC), Taylor-series approximations of portfolio risk, spline
  and polynomial interpolation (fit a yield curve).
- **Topics:** Monte Carlo, quasi-Monte Carlo, variance reduction, LHS,
  importance sampling, numerical integration, Taylor approximations,
  polynomial interpolation, spline fitting.
- **Promote:** `qfin/simulation/` (mc.py, qmc.py, variance_reduction.py,
  sampling.py, interpolate.py).
- **Deliverable:** convergence study (plain MC vs. Sobol vs. LHS) on a VaR/CVaR
  estimate for the project 07 portfolio, plus a fitted yield curve.
- **Feeds into:** MC-VaR in 07; simulation engine for stress tests and the
  capstone.

### ☐ Project 11 — Backtesting & Performance Attribution
**Anchor question:** *Where did my returns actually come from?*

- Build a small vectorized backtester from scratch, then compare with
  vectorbt. Walk-forward testing (train on the past, trade the future) as the
  honest alternative to single-shot optimization. Performance attribution:
  return decomposition (allocation vs. selection effects), per-asset and
  per-factor contribution analysis, benchmark comparison.
- **Topics:** backtesting, return decomposition, contribution analysis,
  performance attribution, benchmarking, walk-forward validation, backtesting
  statistics (Sharpe/Sortino/Calmar in live-like conditions), data snooping
  pitfalls.
- **Promote:** `qfin/backtest/` (engine.py, attribution.py, walk_forward.py).
- **Deliverable:** a walk-forward backtest of the project 09 best strategy,
  with a full attribution report explaining where every basis point came from.
- **Feeds into:** the capstone's evaluation harness is this project.

### ☐ Project 12 — Machine Learning for Portfolios
**Anchor question:** *Can a model pick better weights than me?*

- PCA for dimensionality reduction (and its link to factor models), K-means
  clustering for asset grouping (feed the clusters into HRP), random forest /
  gradient boosting for feature selection and return prediction, ensemble
  methods, hyperparameter tuning, and the crucial topic of time-series
  cross-validation (why k-fold lies to you). Neural networks for allocation
  and a small RL agent (Q-learning or a simple policy gradient) as a bonus
  module.
- **Topics:** PCA, K-means clustering, random forest & gradient boosting,
  ensemble methods, hyperparameter tuning, cross-validation for time series,
  neural networks for allocation, reinforcement learning (optional/bonus).
- **Promote:** `qfin/ml/` (clustering.py, pca_utils.py, feature_selection.py).
- **Deliverable:** an ML-suggested allocation vs. the project 09 classical
  baselines, evaluated with the project 11 walk-forward harness.
- **Feeds into:** ML signal layer of the capstone (optional).

### ☐ Project 13 — Scale & Performance
**Anchor question:** *Can this run on 1,000 assets before my coffee cools?*

- Scale MVO and the risk model to a 1000+ asset universe: sparse covariance
  structures, Cholesky and conditioning (numerical stability), floating-point
  precision and rounding errors, memory management, parallelization with
  joblib/multiprocessing, optional GPU experiment with CuPy, solver benchmark
  at scale, complexity analysis of every algorithm in `qfin/`.
- **Topics:** large-scale optimization, sparse matrix operations, parallel
  computing & GPU acceleration, numerical stability, precision & rounding
  errors, memory management, algorithm complexity & efficiency.
- **Promote:** `qfin/optimization/scale.py`, `qfin/risk/large_cov.py`.
- **Deliverable:** a benchmark report: runtime and memory for 10 → 100 → 1000
  assets, before and after optimization.
- **Feeds into:** the capstone if you want it to run at scale.

### ☐ Project 14 — Capstone: A Complete Quant Strategy
**Anchor question:** *Can I build a strategy pipeline end-to-end — and defend every number in the report?*

The full pipeline, composed entirely from `qfin/`:

1. **Universe & data** (00) — cached prices and fundamentals.
2. **Signals** — value scores from the DCF engine (03), factor scores (02),
   momentum, optional ML signals (12).
3. **Views & allocation** — Black-Litterman or HRP with risk parity
   comparison (09).
4. **Risk model** — shrunk factor covariance, GARCH forecasts (07).
5. **Optimization** — constrained max-Sharpe or CVaR objective with turnover
   and cost constraints (06 + 08).
6. **Backtest** — walk-forward with transaction costs (11).
7. **Report** — attribution (11), metrics table (01), stress tests (07),
   regime context (04).

**Deliverable:** a repository-worthy strategy package + a research report where
every number traces back to a `qfin/` function you wrote and tested.

---

## Suggested pace

Roughly **2 weeks per project ≈ 7 months** at a sustainable pace. Some projects
are naturally lighter (00, 02) and some heavier (07, 09, 14), so it will
average out. The sequence is designed in order — skip at your own risk, since
each project imports the previous ones. The one reorder that works: 03 (DCF)
and 04 (time series) are independent of each other and can swap.

## Progress tracker

- [ ] 00 — Data Toolkit
- [ ] 01 — Risk & Return Metrics
- [ ] 02 — CAPM, Beta/Alpha & Factor Models
- [ ] 03 — DCF & Valuation Multiples
- [ ] 04 — Time Series, Kalman Filters & Regimes
- [ ] 05 — Mean-Variance & the Efficient Frontier
- [ ] 06 — Solver & Optimization Lab
- [ ] 07 — Risk Modeling
- [ ] 08 — Constraints & Real-World Frictions
- [ ] 09 — Advanced Asset Allocation
- [ ] 10 — Monte Carlo & Numerical Methods
- [ ] 11 — Backtesting & Performance Attribution
- [ ] 12 — Machine Learning for Portfolios
- [ ] 13 — Scale & Performance
- [ ] 14 — Capstone: Complete Quant Strategy

---

## Appendix — Topic Coverage Matrix

Every requested topic, mapped to the project(s) where it is practiced.

### Portfolio Optimization Fundamentals
| Topic | Project(s) |
|---|---|
| Modern Portfolio Theory (Markowitz) | 05, 14 |
| Mean-Variance Optimization | 05, 08 |
| Risk-Return Tradeoff | 01, 05 |
| Efficient Frontier | 05, 08 |
| Capital Asset Pricing Model (CAPM) | 01, 02, 03 (WACC) |
| Sharpe Ratio & Risk-Adjusted Returns | 01, 09, 11 |
| Maximum Sharpe Ratio Portfolio | 05, 06, 14 |
| Minimum Variance Portfolio | 05, 07 |
| Risk Parity Strategies | 09, 14 |

### Advanced Optimization Techniques
| Topic | Project(s) |
|---|---|
| Convex Optimization | 05, 06 |
| Quadratic Programming (QP) | 05, 06, 08 |
| Linear Programming (LP) | 06 |
| Semidefinite Programming (SDP) | 06 |
| Second-Order Cone Programming (SOCP) | 06, 08 (CVaR objective) |
| Lagrangian Relaxation | 06 |
| Gradient Descent & SGD | 06 |
| Interior Point Methods | 06, 13 (solver behavior at scale) |
| Sequential Least Squares Programming (SLSQP) | 05, 06 |

### Asset Allocation Strategies
| Topic | Project(s) |
|---|---|
| Strategic Asset Allocation | 09 |
| Tactical Asset Allocation | 09, 14 |
| Dynamic Asset Allocation | 04, 09, 14 |
| Factor-Based Allocation | 02, 09, 14 |
| Risk Factor Decomposition | 02, 11 (attribution) |
| Smart Beta Strategies | 09 |
| Equal Weight vs Market Cap Weight | 01, 05, 09 |
| Black-Litterman Model | 09, 14 (views from 03's DCF) |
| Hierarchical Risk Parity (HRP) | 09, 12 (clusters feed HRP) |

### Approximation & Numerical Methods
| Topic | Project(s) |
|---|---|
| Monte Carlo Simulation | 10, 07 (MC-VaR) |
| Quasi-Monte Carlo Methods | 10 |
| Variance Reduction Techniques | 10 |
| Latin Hypercube Sampling | 10 |
| Importance Sampling | 10 |
| Numerical Integration | 10 |
| Taylor Series Approximations | 03 (sensitivities), 10 |
| Polynomial Interpolation | 10 |
| Spline Fitting | 10 (yield curve) |

### Constraints & Real-World Considerations
| Topic | Project(s) |
|---|---|
| Box Constraints (position limits) | 08, 14 |
| Linear Constraints (sum to 100%) | 05, 08 |
| Turnover Constraints | 08, 11, 14 |
| Transaction Costs Modeling | 08, 11, 14 |
| Slippage Estimation | 08, 11 |
| Cardinality Constraints | 08 |
| Sector/Industry Constraints | 08, 14 |
| Short-Sale Restrictions | 05 (long-only), 08 |
| Minimum Position Sizes | 08 |

### Risk Modeling & Estimation
| Topic | Project(s) |
|---|---|
| Covariance Matrix Estimation | 05, 07, 13 |
| Correlation Matrices | 07 |
| Shrinkage Estimators (Ledoit-Wolf) | 07, 14 |
| Factor Models (Fama-French, APT) | 02, 07 (factor covariance) |
| Value at Risk (VaR) | 01 (intro), 07, 10 |
| Conditional Value at Risk (CVaR) | 07, 08, 14 |
| Volatility Forecasting (GARCH, EWMA) | 07, 14 |
| Tail Risk Estimation | 07, 10 |
| Stress Testing | 07, 14 |

### Machine Learning & Modern Approaches
| Topic | Project(s) |
|---|---|
| Neural Networks for Portfolio Optimization | 12 |
| Reinforcement Learning for Asset Allocation | 12 (optional/bonus) |
| Random Forest & Gradient Boosting for Feature Selection | 12 |
| Clustering Algorithms (K-means for asset grouping) | 12 (feeds HRP in 09) |
| Dimensionality Reduction (PCA) | 12, 02 (link to factor models) |
| Ensemble Methods | 12 |
| Hyperparameter Tuning | 12 |
| Cross-Validation Strategies | 11 (walk-forward), 12 |

### Implementation & Tools
| Topic | Project(s) |
|---|---|
| Libraries: NumPy, SciPy, Pandas | all |
| Libraries: PyPortfolioOpt, CVXPY, Pyomo | 05 (validation), 06, 08 |
| Solvers: CPLEX, Gurobi, MOSEK, ECOS, SCS | 06, 13 |
| Backtesting: Backtrader/Zipline/VectorBT | 11 (vectorbt vs. hand-rolled) |
| Data Handling: real-time vs historical, benchmarks | 00 |
| Performance Attribution: decomposition, contribution | 11, 14 |

### Computational Challenges
| Topic | Project(s) |
|---|---|
| Large-Scale Optimization (1000+ assets) | 13, 14 |
| Sparse Matrix Operations | 13 |
| Parallel Computing & GPU Acceleration | 13 |
| Algorithm Complexity & Efficiency | 06, 13 |
| Numerical Stability | 13, 07 (covariance conditioning) |
| Precision & Rounding Errors | 13 |
| Memory Management | 13 |

### Statistical & Econometric Topics
| Topic | Project(s) |
|---|---|
| Time Series Analysis & Stationarity | 04 |
| Regime Switching Models | 04, 14 |
| Kalman Filtering | 04 (dynamic beta) |
| Autoregressive Models (AR, ARIMA) | 04 |
| Hypothesis Testing | 01, 02 (alpha significance) |
| Backtesting Statistics (Sharpe, Sortino, Calmar) | 01, 11 |
| Performance Benchmarking | 01, 11 |

### Added by request (not in the original list)
| Topic | Project(s) |
|---|---|
| Beta & Alpha analysis | 01 (OLS + rolling), 02 (factor-adjusted), 04 (Kalman) |
| Valuation Multipliers (P/E, EV/EBITDA, P/B, PEG...) | 03 |
| DCF in Python | 03 |