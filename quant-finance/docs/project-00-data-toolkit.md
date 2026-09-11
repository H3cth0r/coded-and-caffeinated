# Project 00 — Data Toolkit: Implementation Plan

> **Anchor question:** *How do I get clean, cached price data for any ticker?*
>
> **Status:** planning — implementation not started.
> **Parent doc:** [index.md](index.md) · Course overview in the repo [README](../README.md).

This document is the implementation plan for Project 00. It captures the design
decisions, the repository skeleton, and the module-by-module contract for the
`qfin/data/` package before any code is written.

---

## 1. Scope

Project 00 builds the first promoted package of the `qfin` library:

- The `PriceProvider` abstraction and two implementations
  (`YFinanceProvider`, `MassiveProvider`).
- A caching layer (`CachedProvider`) so every price series is downloaded
  exactly once.
- Transformation helpers: simple/log returns, alignment, resampling, and a
  price-validation utility.
- The deliverable: a single command that fetches and caches SPY + ~20 stocks,
  idempotent (a second run performs no network calls).

**Out of scope:** fundamentals data (project 03's need — the provider naming
just leaves room for a future `FundamentalsProvider`), real-time data, the
self-computed adjustment factors (see §6, Known simplifications).

---

## 2. Locked design decisions

Decided during planning (2026-09-11):

| # | Decision | Choice | Rationale / consequence |
|---|----------|--------|-------------------------|
| 1 | Massive API access | **Free tier**: 5 requests/minute, **end-of-day data only**, up to 2 years of historical data (paid plans start at $29/month and remove request limits) | `MassiveProvider` gets throttling + retry-with-backoff baked in. Batch fetches rely on the cache: one download per ticker means we rarely hit limits. See §6.5 for the full free-tier operating rules and their consequences. |
| 2 | `get_prices` contract | **Wide close prices** — `DataFrame[dates × tickers]`, one field per call | Simplest contract for the metrics work that dominates projects 01+. Full OHLCV is still available per ticker via `get_history`. |
| 3 | Batch failure mode | **Warn and continue**, via a batch helper that wraps strict providers | Providers raise typed errors; the tolerant wrapper collects failures into a `FetchReport`. Strict components + tolerant wrapper is easier to reason about than providers that half-fail. |
| 4 | Cache location | Repo-local `.data/` (gitignored), overridable via `QFIN_CACHE_DIR` | Inspecting the cache by hand is part of this project's learning. Can move to `~/.cache/qfin` later via the env var. |
| 5 | Adjusted-price policy | Cache stores **both** `close` (raw) and `adj_close`; `adjusted=True` is the default | Raw close = "the price I would have seen that day" (valuation, comps). adj_close = "the price used for returns" (metrics). Returns helpers refuse to compute on raw close. |
| 6 | API key variable | `POLYGON_API_KEY` (standard SDK name), loaded from a gitignored `.env` | Massive is formerly Polygon; the SDK and all documentation use this name. Don't invent a variable the SDK won't auto-read. |
| 7 | Python pin | 3.12 | Safe default; matches `requires-python` and `ruff target-version`. |
| 8 | Data freshness policy | Refetch full ticker history when the requested range touches recent data (~5 trading days from cached max date) | Adjusted prices are retroactively revised after new dividends/splits; a forever-sliced cache is silently stale. One API call, cheap. See §6. |

---

## 3. Implementation sequence

1. **Repo skeleton** — `pyproject.toml`, venv, dirs, git init (§4).
2. **Notebook prototype** — raw API exploration, no library code (§5).
3. **Build `qfin/data/`** in dependency order:
   `base.py` → `yfinance.py` → `cache.py` → `transforms.py` → `massive.py`
   → `universe.py` (§6).
4. **Tests without network** — synthetic data for transforms; recorded
   fixtures for providers so `CachedProvider` tests run fully offline (§7).
5. **Deliverable** — `scripts/fetch_universe.py` + project README notes on
   real-time vs. historical data and benchmark selection (§8).
6. **Promote & tag** — sanity tests green, `git tag v0.1-data` (§9).

---

## 4. Step 0 — Repo skeleton

### 4.1 Directory tree

```
quant-finance/
├── README.md                       ← course plan; gains a Documentation pointer
├── pyproject.toml
├── .gitignore
├── .env                            ← gitignored; POLYGON_API_KEY
├── .data/                          ← gitignored; local parquet cache
│
├── qfin/                           ← THE LIBRARY
│   ├── __init__.py                 ← version only, for now
│   └── data/                       ← the project-00 package
│       ├── __init__.py             ← public API re-exports
│       ├── base.py                 ← Protocol, errors, FetchReport, fetch_universe
│       ├── cache.py                ← CachedProvider + disk layout
│       ├── transforms.py           ← returns, alignment, resampling, validation
│       ├── universe.py             ← default ticker lists, benchmarks
│       ├── yfinance.py             ← YFinanceProvider
│       └── massive.py              ← MassiveProvider (throttled)
│
├── tests/
│   ├── conftest.py                 ← offline cache dir, fake provider fixtures
│   ├── test_transforms.py
│   ├── test_cache.py
│   └── test_providers.py
│
├── projects/
│   └── 00-data-toolkit/
│       ├── README.md               ← anchor question, decisions log, learnings
│       ├── notebooks/
│       │   └── 01-provider-exploration.ipynb
│       └── scripts/
│           └── fetch_universe.py   ← the deliverable
│
└── docs/
    ├── index.md
    └── project-00-data-toolkit.md  ← this file
```

**Naming caveat:** `qfin/data/yfinance.py` shares its name with the
`yfinance` package. This is safe under Python 3's absolute imports *as long as*
nothing is ever executed with `qfin/data/` as the working directory (scripts
run from repo root; imports go through the editable install). The zero-risk
variant is `yfinance_provider.py` / `massive_provider.py` — functionally
identical, decided at implementation time.

### 4.2 pyproject.toml

Deliberately minimal — dependencies join when a project needs them.

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "qfin"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=16",            # parquet cache
    "yfinance>=0.2.40",
    "polygon-api-client",      # Massive (formerly Polygon) SDK
    "python-dotenv",           # load .env for the API key
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "ruff", "jupyterlab", "ipykernel"]
# scipy / statsmodels / matplotlib / cvxpy... join when projects 01+ need them

[tool.setuptools.packages.find]
include = ["qfin*"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

### 4.3 Setup commands

```bash
cd quant-finance
git init
uv venv --python 3.12
uv pip install -e ".[dev]"     # editable; everyday installs can use plain pip
cp .env.example .env           # paste the Massive/Polygon key
```

`.gitignore`: `.venv/`, `.data/`, `.env`, `__pycache__/`, `.pytest_cache/`,
`.ipynb_checkpoints/`, `*.egg-info/`.

---

## 5. Step 1 — Notebook prototype (the messy phase)

`projects/00-data-toolkit/notebooks/01-provider-exploration.ipynb` — no library
code, exploration only. Its goals:

1. **Adjusted vs. unadjusted prices.** Download both series for a stock with a
   visible split/dividend history; plot them. The divergence *is* the lesson —
   this is the classic silent bug in every downstream metrics computation.
2. **Raw response shapes.** Inspect what yfinance and the Massive API actually
   return (column MultiIndexes, timezones, units) so the providers' normalizing
   code is written from understanding, not guesswork.
3. **Calendar mismatch table.** Which holidays each provider covers, timezone
   handling, missing-day behavior — the inputs to `align_prices`'s policies.
4. **Rate-limit reality check.** Confirm Massive free-tier behavior (5 req/min,
   history depth) so the throttle constants aren't guesses.
5. **Why SPY.** A short written note on benchmark selection: liquidity,
   investability, representativeness — feeding `universe.py` and the
   benchmark-selection topic.

---

## 6. Step 2 — `qfin/data/` module design

### 6.1 The canonical data model

Every provider's `get_history(ticker)` returns this exact shape — and it is
what gets cached:

```
DataFrame, DatetimeIndex (naive, exchange calendar, ascending, unique)
columns: open | high | low | close | volume | adj_close
```

`get_prices` is the convenience layer on top:

```python
def get_prices(self, tickers, start, end, *,
               field: str = "close",   # open|high|low|close|volume|adj_close
               adjusted: bool = True,  # True → adj_close, False → close
              ) -> pd.DataFrame         # wide: index=dates, columns=tickers
```

**Why keep both `close` and `adj_close`:** raw close answers "what was the
price on that day" (valuation, comps — project 03); adj_close answers "what
price do returns come from" (metrics everywhere). The returns helpers refuse
to compute on raw close unless explicitly overridden — silent-bug insurance.

### 6.2 `base.py` — contract, errors, batch helper

```python
class PriceProvider(Protocol):
    def get_history(self, ticker: str, start=None, end=None) -> pd.DataFrame: ...
    def get_prices(self, tickers, start, end, *, field="close", adjusted=True) -> pd.DataFrame: ...

class ProviderError(Exception): ...
class TickerNotFound(ProviderError): ...
class HistoryLimitError(ProviderError): ...   # Massive free-tier depth

@dataclass
class FetchReport:
    ok: list[str]
    failed: dict[str, str]   # ticker -> reason

def fetch_universe(provider, tickers, start, end, *, field="close")
    -> tuple[pd.DataFrame, FetchReport]: ...
```

`fetch_universe` loops tickers, catches `ProviderError` per ticker, collects
failures into the report, and emits warnings. It is a **free function, not part
of the Protocol** — the Protocol stays stateless. Providers are strict; the
wrapper is tolerant (decision #3).

### 6.3 `cache.py` — CachedProvider

`CachedProvider(provider)` wraps any provider transparently.

**Disk layout** (repo-local `.data/`, decision #4):

```
.data/prices/{provider}/{TICKER}.parquet    # e.g. .data/prices/yfinance/AAPL.parquet
```

One file per ticker holding the canonical full-history frame.

**Read logic:**

```
read(ticker, start, end):
    df = load parquet or None
    if df is None                                  → miss: fetch full history, write, slice
    elif end > df.index.max() - freshness buffer   → stale: refetch full history, rewrite, slice
    else                                           → hit: local slice only
```

**The freshness rule (decision #8) — the subtle point of this project:**
adjusted prices are *retroactively revised*. A new dividend or split rewrites
the entire historical `adj_close` series upstream, so a cache that slices
yesterday's download forever is silently wrong. Policy: if the requested range
touches data within ~5 trading days of the cached max date, refetch the
ticker's full history and rewrite the file (one API call — cheap). Local
slicing is trusted only for ranges fully in the past.

**Environment override:** `QFIN_CACHE_DIR` replaces `.data/` as the cache root.

### 6.4 `yfinance.py` — built first

Wraps `yf.download(ticker, auto_adjust=False, actions=False)`, normalizes the
column MultiIndex, drops the timezone, renames `Adj Close` → `adj_close`, and
returns the canonical frame. Small module (~60 lines); the notebook explains
why those flags matter.

### 6.5 `massive.py` — built last

Same canonical output from
`RESTClient().get_aggs(ticker, 1, "day", start, end)`, plus:

- a **throttle**: minimum-seconds-between-calls gate (free tier ≈ 5 req/min);
  a simple gate, not a token bucket,
- **retry with backoff** on HTTP 429,
- `HistoryLimitError` when the requested start predates free-tier history
  (~2 years) — a typed error so `fetch_universe` can report
  "outside free-tier history" distinctly from "bad ticker".

Built last so throttle/backoff code is written against a stable, already-tested
contract — and so a flaky Massive tier never blocks the library.

#### Massive free-tier operating rules

The free tier's exact constraints and what each one means for this library:

| Constraint | Value | Consequence for `qfin/data/` |
|---|---|---|
| Rate limit | 5 requests/minute | Throttle gate at **≥ 12 seconds between calls**; retry with backoff on 429. First-ever fetch of the 21-ticker universe through Massive alone takes **≥ ~4–5 minutes** — acceptable because the cache means it happens exactly once per ticker. |
| Data frequency | **End-of-day only** (no intraday/minute aggregates) | `MassiveProvider` serves daily bars only; `get_history` returns 1-day aggregates. If intraday ever matters, it is a paid-tier feature — not a library gap. This also confirms there is no real-time data path at this tier (feeds the "real-time vs. historical" topic note in §5/§8). |
| History depth | Up to ~2 years (~504 trading days) | `HistoryLimitError` for earlier starts. Enough daily observations for project 01's metrics, but **structurally short** for later projects (rolling beta windows, Fama-French regressions, walk-forward backtests). |

**Role of each provider, as a consequence of the above:**

- `YFinanceProvider` is the **primary workhorse** — free, effectively
  unlimited history, used by the deliverable script and by all early projects.
- `MassiveProvider` is the **validation implementation** — it exists to prove
  the provider abstraction holds across two genuinely different APIs (the
  implement-from-scratch → use-the-library → compare principle, applied to the
  data layer itself). It is exercised in the notebook and in tests, not in the
  hot path.
- **Upgrade path (not assumed):** $29/month removes request limits. If that
  ever happens, only the throttle constants and `HistoryLimitError` handling
  change — by design, nothing else in `qfin/data/` knows which tier is active.

### 6.6 `transforms.py` — pure, offline-testable functions

```python
simple_returns(prices)          # refuses raw close unless explicitly overridden
log_returns(prices)
align_prices(df, how="inner", ffill=True, limit=None)   # docstring states the ffill trade-off
resample_prices(df, rule="W-FRI" | "ME")                # last price in period
validate_prices(df)             # monotonic unique index, non-positive prices, gap detection
```

`validate_prices` is more than a helper — every later project calls it once
before trusting data.

### 6.7 `universe.py` — deliberately tiny

```python
DEFAULT_UNIVERSE = ["SPY", ...20 liquid names]
BENCHMARKS = {"sp500": "SPY", "nasdaq100": "QQQ", ...}
def get_benchmark(name: str) -> str: ...
```

Its real job is being the meeting point of project 00 and everything after.

### 6.8 `qfin/data/__init__.py` — the public API

Later projects import only from the top level:

```python
from qfin.data import (
    PriceProvider, CachedProvider, YFinanceProvider, MassiveProvider,
    get_prices, fetch_universe, simple_returns, log_returns, ...
)
```

---

## 7. Step 3 — Tests without network

| Suite | Strategy |
|-------|----------|
| `test_transforms.py` | Synthetic data with hand-computed expected values (a known split/dividend series for adj_close checks). |
| `test_cache.py` | A `FakeProvider` fixture serving canned frames; verifies miss/stale/hit behavior, the freshness refetch, and `QFIN_CACHE_DIR` isolation. |
| `test_providers.py` | Recorded real API responses stored as fixtures; the provider modules are tested as pure normalizers (raw payload → canonical frame). No network in CI or local runs. |

---

## 8. Step 4 — The deliverable

`projects/00-data-toolkit/scripts/fetch_universe.py`:

- Fetches `DEFAULT_UNIVERSE` (SPY + 20 stocks) through `CachedProvider` with
  `YFinanceProvider` underneath — Massive's free-tier rate limit makes it the
  wrong workhorse for a 21-ticker first fetch (≥ ~4–5 min; §6.5),
- uses `fetch_universe`'s warn-and-continue semantics; exits nonzero only if
  *every* ticker fails,
- is **idempotent**: a second run performs zero network calls,
- prints a per-ticker status summary (cached / fetched / failed + reason).

The project README (`projects/00-data-toolkit/README.md`) also records the two
written-topic notes from §5 (real-time vs. historical data, benchmark
selection) and a running decisions log.

---

## 9. Step 5 — Promote & tag

- [ ] All tests pass offline (`pytest`, no network).
- [ ] Deliverable script runs twice; second run is cache-only.
- [ ] `qfin/data/` modules have docstrings stating the contracts above.
- [ ] Commit and tag **`v0.1-data`** — the library's first visible growth step.

---

## 10. Known simplifications (honest-scopes note)

Recorded here and to be copied into the project README:

1. **Adjustment factors are trusted, not computed.** The "proper" fix for the
   retroactive-revision problem is storing raw close + split/dividend events
   and computing adjustment factors ourselves. Explicitly deferred — the
   freshness-refetch policy (§6.3) is the pragmatic approximation.
2. **No real-time data path.** Providers are historical-only for now;
   "real-time vs. historical" is covered as a topic, not an implementation.
   Confirmed structural at the Massive free tier (EOD-only — §6.5).
3. **Single exchange calendar assumption.** `align_prices` assumes US equity
   sessions; multi-market alignment is a later problem.
4. **Fundamentals are out of scope**, but provider naming leaves room for a
   future `FundamentalsProvider` beside `PriceProvider` (project 03).
5. **Massive free tier caps history at ~2 years.** Long-history work in later
   projects (rolling windows, factor regressions, walk-forward) will use
   yfinance depth; Massive serves the abstraction-validation role (§6.5).
   Accepted as a tier limitation, not a library defect.