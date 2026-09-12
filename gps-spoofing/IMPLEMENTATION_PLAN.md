# Implementation Plan — Build Your Own GPS Spoofing Dataset & Detector

*The companion to `README.md`. Same friendly tone, one new job: turn everything we learned (and everything the paper did wrong) into a build plan you can actually follow.*

---

## Three ground rules before we start

1. **No hardware needed. Ever.** Every phase is pure software. The GPS "knock patterns" (PRN codes) are public math, the "satellites" can be a made-up-but-geometrically-consistent constellation, and the noise is just random numbers. The only place *real* signals show up is final validation — and even there we **download** the public TexHex recordings instead of recording our own. Any step that mentions real antennas is clearly marked **[OPTIONAL HARDWARE]** and can be skipped forever.
2. **Every phase ends with a picture.** Plots are our "trust but verify" layer. A number without a picture is a rumor. Each phase below lists exactly which plots you should look at before moving on.
3. **The iron rule (the paper broke it, we don't):** every feature is *computed from the actual waveform*. No feature is ever drawn from a probability distribution. If the waveform doesn't produce the pattern, the feature must not show it.

---

## What we copy vs. what we fix

Every mistake we found in the paper becomes a design rule for us:

| The paper did this | We will do this instead | Why |
|---|---|---|
| Drew Features 1–2 from **probability distributions** | Compute **every feature from the actual waveform** | If you *invent* the vital signs, the ML learns your imagination, not physics |
| ~70/30 class split, no balancing | Balanced-ish classes + **class weights** | A model that says "all fine" gets 70% for free |
| Every model at its **factory-default threshold** | Tune each model's threshold on its **ROC curve** | We want "catch the most attacks," not "best mixed score" |
| Picked the winner by **accuracy** | Pick by **Pd at a Pfa budget** | A missed attack is a disaster; a false alarm is a cheap retry |
| **One** train/test split | **Repeated cross-validation** + confidence intervals | With one test, you can't tell "winner" from "lucky" |
| 1,000 samples | 10,000+ samples | More data = less guessing |
| No **dumb baseline** (logistic regression) | Always train a dumb baseline first | If the fancy models can't beat the dumb one, the fancy models are decoration |
| "Energy-efficient" claimed, **never measured** | Report inference time per decision | Unmeasured claims are just vibes |
| Ground-based spoofer assumption | Include **airborne** and **swarm** attack variants | Test the threat that actually breaks the assumption |
| Pure AWGN channel | AWGN → multipath → (downloaded) **real noise** | Real GPS lives in a noisy world, not a math textbook |
| One simulator only | Train on one generator, **test on another** + **TexHex** | Proves we learned physics, not the simulator's fingerprints |

---

## Phase 0 — Toolbox setup (1 day)

This is just "npm install" for radio stuff — all free, all software.

```bash
cd gps-spoofing
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy scikit-learn matplotlib pandas skyfield
```

What each tool is, in kid words:

- **numpy / scipy** — the Lego bricks of signal math (waves, filtering, matrix tricks)
- **scikit-learn** — the shelf where all the classifiers already exist, ready-made
- **matplotlib** — draws every picture in this plan
- **pandas** — the spreadsheet for your dataset
- **skyfield** — tells you where each satellite *actually is* in the sky, using real NASA data (nice-to-have, not required — see Step 1.2)

**Download (still no hardware, just files):** the public **TexHex** dataset (UT Austin Radionavigation Lab) — real RF recordings of clean GPS *and* real spoofing. Keep it untouched until Phase 6. It's our final exam.

**Also try to obtain:** reference [15] of the paper — *"Realistic GPS spoofing dataset generation with RF augmentation for adversarial environment simulation"* (DISCOVER 2025, same authors). It's the companion paper where the original dataset recipe actually lives. Read it before Phase 1 — if their RF-augmentation details are better than ours anywhere, steal shamelessly and cite.

**Suggested repo layout:**

```
gps-spoofing/
├── README.md                  ← the explainer
├── IMPLEMENTATION_PLAN.md     ← you are here
├── data/
│   ├── texhex/                ← downloaded real recordings (Phase 6 only)
│   └── processed/             ← our generated datasets (parquet/npz + manifest.json)
├── src/
│   ├── signals/               ← Phase 1: authentic signal generation
│   ├── attacks/               ← Phase 2: spoofers
│   ├── receiver/              ← Phase 3: array + MUSIC + features
│   ├── dataset/               ← Phase 4: assembly + splits
│   ├── models/                ← Phase 5: training
│   └── evaluation/            ← Phase 6: metrics + plots
├── notebooks/                 ← your playground
└── tests/                     ← sanity checks (see "Traps")
```

---

## Phase 1 — Build an "authentic" GPS signal from scratch (1–2 weeks)

**The happy thought that makes this possible:** every part of a GPS signal is **public**. The knock pattern of every satellite is printed in a public document (the GPS ICD). It's like a spy code where the codebook is on the internet — the secrecy was never the point; the *precision* is.

### Step 1.1 — Generate the PRN codes (1 day)

Each satellite's code: 1,023 "chips" (±1 values), repeating every 1 millisecond, at 1.023 million chips/second. The recipe (a **Gold sequence**): make two special marching rows of 0/1 (shift registers) and combine them — "same? → −1, different? → +1". The ICD is just a lookup table of starting positions per satellite; implement it once.

```python
# pseudocode — the whole "personal knock pattern" generator
def gold_code(prn):  # prn = satellite number 1..32
    g1 = shift_register(taps=[...], init=[...])   # from the ICD table
    g2 = shift_register(taps=[...], delay=prn)
    return [2*a - b for a, b in zip(g1, g2)]      # → ±1 chips
```

**Sanity check (non-negotiable):** shift the code against itself by 1 chip → match should be *terrible* (~half disagree). Align perfectly → match should be *perfect*. Plot the correlation for offsets −10…+10 chips: one sharp spike, everything else near zero.

> 📈 **Plots this phase:** the correlation-vs-shift plot (one sharp spike at zero = the engine works).

### Step 1.2 — Where are the satellites? (1 day, deliberately simplified first)

**Simplified version (do this first):** invent a constellation — 8–12 "satellites" at plausible elevations (20°–85°) and azimuths. The detector never needs them to be *real*; it needs them to be **geometrically consistent** — delays, Doppler, and arrival angles must all agree with each other, like a well-written lie.

**Upgraded version (when you want it):** `skyfield` + a real ephemeris file → actual satellite positions. Same code downstream either way.

> 📈 **Plots this phase:** a sky-map (polar plot) of your constellation — satellites scattered across elevations/azimuths, the way a real sky looks.

### Step 1.3 — The navigation message (½ day)

A slow stream of data bits (50 bits/second) with time stamps and "here's where I am." Simplified version: a repeating pattern of known bits — fine, because our features don't depend on the message *content*, just its timing.

### Step 1.4 — Assemble the signal (the paper's Eq. 1, our code)

```
s_k(t) = √power × code(t − delay_k) × bits(t) × carrier(t)
```

- Upsample chips to 4.092 MHz sample rate (4× the chip rate — keeps everything aligned)
- `carrier(t)` = sine at an IF (intermediate frequency — a shifted-down copy of 1575.42 MHz; simulate at ~4–10 MHz instead of 1.5 GHz so your laptop doesn't cry)
- Each satellite gets its own code, delay, Doppler (Hz), and power (low-elevation satellites weaker, C/N₀ ≈ 40–50 dB-Hz)

**The Doppler invariant — memorize this.** Doppler isn't random: satellite approaching → frequency up, receding → down, and the *amount* is fixed by geometry (about ±4 kHz max). Every satellite gets *its own correct* Doppler. Later this exact invariant is what fakes can't cheat.

> 📈 **Plots this phase:** the real part of the composite signal (looks like noise — *that's the point*, it's hidden below the static); the correlation peak per satellite (clean spikes again).

### Step 1.5 — The channel (static → weather → real street)

Three difficulty levels; keep all three:

1. **AWGN** — pure static. Easy. Matches the paper. Do it first.
2. **Multipath** — echoes: 1–3 delayed, weakened copies per satellite (bounced off buildings). *This is what makes real signals flicker — and flicker is one of our 5 features, so the authentic class must contain it.*
3. **Real noise — no hardware edition:** extract noise segments from the **downloaded TexHex clean recordings** and mix them in. Real noise, zero antennas.

> 📈 **Plots this phase:** signal power over time — with multipath it should visibly *flicker* (this flicker is a feature we'll exploit later, so verify it exists now).

### Step 1.6 — The antenna array (the paper's Eq. 4–5)

Simulate **M = 4 antennas in a row, spacing d = λ/2**. A signal arriving from angle θ gives each antenna's copy a phase shift: `Δφ = (2πd/λ)·sin(θ)`. The **steering vector** `a(θ) = [1, e^(jΔφ), e^(j2Δφ), e^(j3Δφ)]` is the fingerprint of angle θ. ~15 lines of numpy.

⚠️ **Trap:** keep d = λ/2 exactly. A bigger spacing creates *spatial aliasing* — two different angles producing identical fingerprints, i.e., directions that are impossible to tell apart.

> 📈 **Plots this phase:** phase difference vs. angle curve (smooth S-shape — each angle a unique fingerprint).

---

## Phase 2 — The attacks (1 week)

Each attack = a function that takes the authentic waveform and returns a *poisoned* one. Ordered easy → nightmare:

### Attack A — Meaconing (easy mode, the paper's Eq. 6)

Record the authentic scene, play it back with added delay τ and a power boost:

```python
spoofed = gain * shift(authentic_signal, delay_tau)
```

Every pseudorange becomes *true + c·τ*. Sweep τ (a microsecond of delay ≈ hundreds of meters of displacement) and gain.

### Attack B — Synchronous generative (medium mode, Eq. 7)

1. Compute the pseudoranges for a *fake target location*
2. Generate matching signals with correct per-satellite Doppler
3. Crossfade the receiver's lock: `α(t)` ramps 0 → 1 over tens of seconds, the fake's code phase sliding gradually — never jumping
4. At the end, raise the spoofer's power

The subtlety: during the drift, real + fake coexist and the receiver's tracking loop sits on the *merged* correlation peak. The simulation must reproduce that merge, or the drift looks fake.

### Attack C — Airborne spoofer (the drone)

Same as B, but the spoofer's angle θ is high elevation (60°+), so it looks sky-like. Purpose: prove our detector doesn't secretly rely on "spoofers come from below."

### Attack D — Swarm (hard mode, stretch goal)

2–3 transmitters, each broadcasting a different subset of fake satellites from different angles. This is the attack the paper can't fully handle — the class that keeps our detector honest about its limits.

> 📈 **Plots this phase:** position traces — the receiver's reported position over time for each attack (meaconing: a rigid jump; synchronous: a smooth slow drift; those two shapes *are* the attacks, visible to the naked eye).

---

## Phase 3 — Receiver + feature extraction (1–2 weeks)

### Step 3.1 — Receiver front-end

Correlate against each satellite's code replica → code phase + Doppler → pseudoranges. (Simplified fine; full tracking loops are a later rabbit hole.)

### Step 3.2 — Array processing

- Covariance: `R = (X @ X.conj().T) / N` over N snapshots
- **MUSIC:** eigen-decompose R (numpy `np.linalg.eigh` — one line), take the smallest eigenvalues' vectors as the noise subspace, scan all angles θ with Eq. 9 → the spike graph. Peak-pick with `scipy.signal.find_peaks`.

⚠️ **The M−1 rule (why "number of peaks" ≠ "number of satellites"):** a 4-element array can resolve **at most M−1 = 3 distinct directions** — MUSIC's signal subspace can hold at most 3 sources before it runs out of room. With 8–12 satellites in the sky, most directions merge into fewer peaks. So Feature 1 counts *resolvable clusters*, not satellites — which is exactly why the paper's threshold for this feature is just 2.00. **Document your rule once** (prominence threshold + which elevation subset feeds the array, if any) and apply it identically to both classes, or the feature means different things for authentic vs. spoofed samples — a silent label-flipper.

### Step 3.3 — The 5 features (all computed, zero invented)

| # | Feature | Computed how |
|---|---|---|
| 1 | Number of MUSIC peaks | peak-pick the real spectrum (height/prominence rule — decide *once*, document it) |
| 2 | Angular separation | distance between the two most prominent peaks |
| 3 | Mean C/N₀ | per-satellite signal/noise powers, averaged |
| 4 | Variance of C/N₀ | sample variance across satellites (Eq. 11) |
| 5 | Power stability | variance of power over 200 windows ÷ mean (Eq. 12) |

⚠️ **Trap:** peak-picking rules can differ from the paper's. Doesn't matter — *your* features just need to be consistent within your own dataset. Document the rule.

> 📈 **Plots this phase (the money shots):** MUSIC spike-graphs side by side — authentic (peaks scattered across the sky) vs. spoofed (peaks bunched together). If you can see the difference with your own eyes, the ML can learn it. If you can't, fix the simulation before the ML, never after.

---

## Phase 4 — Dataset assembly (1 week)

**The big difference from the paper: big and honest.**

- **Size:** 10,000+ instances (simulating in software costs nothing but electricity)
- **Class balance:** ~50/50 or at least 60/40 authentic/spoofed — and *report the exact counts*
- **Sweep the knobs:** SNR (−15…+15 dB), attack type (A–D), τ, spoofer power, drone altitude, receiver trajectory, constellation geometry
- **Balanced difficulty:** emphasize the hard middle (the 5°–8° ambiguity zone) — but let it *emerge from physics*, don't engineer it artificially like the paper did
- **Splitting (the honest way):**
  - split **by scenario, not by sample** — one "world" (one trajectory + one attack config) stays entirely on one side; prevents "I've seen this exact background before" cheating
  - **hold out** one whole attack type (e.g. never train on Attack D, test on it) — measures generalization to *unseen attacks*, the thing the paper never measured
- **Manifest everything:** each dataset ships a JSON (seeds, ranges, generator version). A dataset without a manifest is a mystery novel with the last chapter torn out.

> 📈 **Plots this phase:** feature histograms per class (5 panels: authentic vs. spoofed overlaid) + a 2D scatter of the two best features. This is where you *see* the ambiguity zone — and spot any dead-on-arrival feature (looking at you, variance of C/N₀) before the ML does.

---

## Phase 5 — Model development (1–2 weeks)

### Step 5.1 — Preprocessing

- Normalize features — fit the scaler on **train only** (never leak test statistics; classic trap)
- Histogram check first: any feature with overlapping-unto-useless distributions gets pruned (fix for the paper's coin-flip feature)

### Step 5.2 — Train the 5 paper models (plus 2 of our own)

| Model | Notes |
|---|---|
| **Logistic Regression** | **the dumb baseline** — every other model must beat it or justify itself. The paper skipped this; we don't |
| KNN (k=5) | same as paper |
| SVM (RBF) | same as paper |
| Naive Bayes | same as paper |
| Decision Tree (≤10 splits) | same as paper — interpretability is a feature, keep it |
| Random Forest (100+) | same as paper |
| Gradient Boosting (HistGradientBoosting) | our addition — usually tops small-feature tabular problems |

- **Class weights** (`class_weight='balanced'`) everywhere supported
- Fixed random seeds; log every hyperparameter
- Bonus honesty: measure **inference time per decision** (the paper claims energy-efficiency with zero numbers; we'll have actual ones)

### Step 5.3 — Evaluation the honest way

For **every** model, compute and *show*:

1. **The full confusion matrix** (TP/FP/TN/FN — printed every time, no exceptions)
2. **The ROC curve** — the whole dial, not one setting
3. **Pd at a Pfa budget**: detection rate when false alarms ≤ 5% (and ≤ 10%)
4. **Repeated cross-validation**: 5 repeats × 5 folds → mean ± std per metric. A gap smaller than the error bars is decoration, not a ranking
5. **The permutation check** — shuffle the labels; a real detector's AUC collapses to ~0.5, a leaky one doesn't. If it doesn't collapse, hunt the leak. This one test catches 90% of dataset bugs.

### Step 5.3.5 — Threshold selection (the step the paper skipped)

On the *training* ROC curve, pick the threshold that **maximizes Pd subject to Pfa ≤ 5%**. Freeze it. *Then* evaluate once on the test set. The threshold is a hyperparameter — tuning it on test data is reading the exam answers before the exam.

> 📈 **Plots this phase:** ROC curves (all models on one plot), confusion-matrix heatmaps (one per model), a DET-style Pd-vs-Pfa curve zoomed to the 0–10% Pfa region (where real deployments live), and the permutation-check AUC histogram (sharp drop to 0.5 = trustworthy).

---

## Phase 6 — The generalization proof (the part that makes it science)

Two tests the paper never ran — **both still require zero hardware**, only downloaded files:

1. **Cross-simulator test:** train on your simulator's data; test on data from a *different* generator (different channel model, or gps-sdr-sim output). Accuracy holds → we learned physics. It craters → we learned the simulator's handwriting.
2. **TexHex test (the final exam):** run the trained detector on the downloaded real recordings — features extracted with *our* code, on *real* RF, containing *real* spoofing. Pass ≈ AUC ≥ 0.9 → the paper's headline claim upgrades from "plausible in simulation" to "demonstrated."

> 📈 **Plots this phase:** side-by-side AUC bars (own simulator vs. cross-simulator vs. TexHex) — the one chart that tells the whole honesty story at a glance.

---

## Phase 7 (stretch) — Mitigation (2–3 weeks, optional)

- **Beamforming:** Eq. (19) directly — ~10 lines of numpy: `w = R⁻¹C (Cᴴ R⁻¹ C)⁻¹ f`, apply via `y = wᴴ x`
- **Position re-estimation:** weighted least squares on trusted pseudoranges + **Doppler consistency rejection** (drop satellites whose measured Doppler differs from geometric Doppler by > ε — the invariant from Step 1.4)
- **Kalman filter:** predict-then-correct smoother (a basic constant-velocity Kalman is ~30 lines)
- **Report honestly:** mitigation performance *with* noise (like Table III's "signal losses" rows) plus confidence intervals. The ideal-condition "complete recovery" is the easy story; the noisy one is the true one.

> 📈 **Plots this phase:** position maps — true position, attacked position, recovered position, on the same chart, noisy case included. The visual is the whole point: a dot flying off 400 km and coming back home.

---

## Phase 8 — **[OPTIONAL HARDWARE]** The real-world layer (only if you're curious)

You never *need* this — everything above is complete without it. But if curiosity wins:

- RTL-SDR (~$30) + GPS antenna → record clean GPS (receiving is legal everywhere)
- Mix synthetic spoofers into the *real recording* (RF augmentation)
- **Never transmit** anything in the L1 band — it's a protected frequency; transmitting requires government authorization. Everything mixes in software. Listening is always fine.

---

## The focus checklist (our 12 fixes, mapped to where they happen)

| # | Paper's weakness | Fixed in | How |
|---|---|---|---|
| 1 | Class imbalance, no weights | Phase 4/5 | balanced classes + `class_weight='balanced'` |
| 2 | No Pd optimization | Phase 5.3.5 | ROC threshold: max Pd s.t. Pfa ≤ 5% |
| 3 | Winner by accuracy | Phase 5.3 | selection metric = Pd @ Pfa budget |
| 4 | One split | Phase 5.3 | 5×5 repeated CV, mean ± std |
| 5 | Small dataset | Phase 4 | 10,000+ samples |
| 6 | Ground-only spoofer | Phase 2 | attacks C (drone) + D (swarm) |
| 7 | Simulation only | Phase 6 | cross-simulator + TexHex validation |
| 8 | Features drawn from distributions | Phase 3 | iron rule: features from waveforms only |
| 9 | Useless feature kept | Phase 5.1 | histogram check first; prune by AUC |
| 10 | No confusion matrix / inconsistent numbers | Phase 5.3 | always print the matrix; re-derive every metric from it |
| 11 | No dumb baseline | Phase 5.2 | logistic regression first; everyone must beat it |
| 12 | Energy claims unmeasured | Phase 5.2 | record inference time per decision |

---

## What we deliberately do NOT solve (and why that's okay)

Being honest up front beats being embarrassed later. Three things we *measure* but don't claim to fix:

1. **Realistic DLL drift (Attack B's hard part).** The synchronous attack only looks real if a genuine tracking loop follows the *merged* correlation peak. Our receiver is deliberately simplified (Step 3.1), so faithfully reproducing the S-curve merge is the hardest open item in this plan. **Fallback:** if the simplified receiver can't reproduce the drift plausibly, keep the attack but *say so in the manifest* — a labeled imperfection is science; an unlabeled one is self-deception.
2. **Swarm mitigation.** With M = 4 antennas we can null roughly 2–3 directions. For Attack D we measure the *detection* limit honestly (held-out class, never trained on), but we do not attempt multi-null beamforming. That's an open research problem, and our job is to know where the cliff is, not to deny it.
3. **Real-time constraints.** We measure inference time per decision, but not full-pipeline latency on live hardware. Enough to back a "more efficient than deep learning" claim with actual numbers — not enough to claim "real-time certified."

---

## Suggested milestones

| Milestone | Definition of done | Roughly |
|---|---|---|
| **M1 — A signal exists** | Correlation with the right PRN finds a clean peak; wrong codes find nothing; the plot shows one spike | week 1–2 |
| **M2 — An array exists** | MUSIC spike-graph spikes at the right angles for a 2-signal scenario | week 3 |
| **M3 — A dataset exists** | 10k labeled samples + manifest; feature histograms look sane | week 4 |
| **M4 — A detector exists** | CV report: confusion matrices, ROC, Pd@Pfa ≤ 5%, mean ± std per model | week 5 |
| **M5 — Generalization proven** | cross-simulator + TexHex chart | week 6+ |
| **M6 (stretch) — Mitigation** | beamformer + Kalman; position map like the paper's Fig. 3, noisy case included | week 8+ |

---

## Traps & gotchas (re-read before every debugging session)

1. **Leakage** — the #1 killer. Normalizer fit on all data, threshold tuned on test, features encoding the label indirectly. When in doubt: does this step see the test data? If yes → stop.
2. **Permutation check** — if shuffling labels doesn't drop AUC to ~0.5, something leaks. No exceptions.
3. **Feature scaling** — KNN and SVM need normalized features; trees don't care. Normalize anyway, consistently.
4. **Complex numbers** — signal processing lives in real + imaginary arithmetic. numpy handles it natively; use `.conj()` where the paper writes `^H`.
5. **Sample-rate discipline** — chip rate 1.023 Mcps must be an exact multiple of your sample rate or codes never align. Pick 4.092 MHz and stay there.
6. **Doppler sign** — toward = positive. Mix it up and every geometry check fails mysteriously.
7. **Aliasing** — antenna spacing stays at λ/2. No exceptions.
8. **Seeds** — one global seed; the dataset must be regenerable. A dataset you can't rebuild is a dataset you can't trust.
9. **Class labels** — one convention (1 = spoofed, like the paper's Eq. 16), written in the manifest. A flipped label silently inverts every metric.
10. **The paper's numbers are approximate** — compare against Table III's *noisy* rows, not the ideal ones.
11. **Plots are checks, not decoration** — every phase's plot list is a gate: if the picture looks wrong, the code is wrong. Look before you pipeline.

---

*Prerequisites: read `README.md` first. The companion paper with the original dataset methodology is reference [15] of `new_final.pdf`. The real-data validation target is TexHex (UT Austin Radionavigation Lab) — public download, no hardware required.*