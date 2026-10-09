# Results

Test period 2025-01-01 -> 2026-09-30 (30,624 half-hour decisions), $1M book, paper accounting (instant fill at the bar close, flat 25 bp) unless a section says otherwise. Method and assumptions: [METHODOLOGY.md](METHODOLOGY.md). All numbers are generated from `artifacts/enh/*.json` by `scripts/make_results_md.py`.

**Context that matters for every number below:** the test window is a bear market (equal-weight buy-and-hold ended at 0.64, BTC at 0.89, only the hindsight best coin made money). Each method row is **one seed, chosen on 2024 validation Sharpe** (2-3 seeds trained), so differences of a few percent are within seed noise.

## Baseline and benchmarks

| strategy | final wealth | ann. return | Sharpe | max DD | turnover/bar | gross | net | weight time-std |
|---|---|---|---|---|---|---|---|---|
| eiie | 0.662 | -21.0% | -0.05 | 68% | 0.0003 | 0.95 | 0.95 | 0.0013 |
| ubah | 0.638 | -22.6% | -0.07 | 68% | 0.0000 | 1.00 | 1.00 | 0.0153 |
| ucrp | 0.574 | -27.2% | -0.13 | 71% | 0.0018 | 1.00 | 1.00 | 0.0000 |
| btc_hold | 0.891 | -6.4% | +0.07 | 54% | 0.0000 | 1.00 | 1.00 | 0.0000 |
| best_stock_hindsight | 1.093 | +5.2% | +0.36 | 60% | 0.0000 | 1.00 | 1.00 | 0.0000 |
| momentum_1d | 0.090 | -74.8% | -1.64 | 96% | 0.0301 | 1.00 | 1.00 | 0.1519 |
| reversal_3h | 0.000 | -100.0% | -35.87 | 100% | 0.6304 | 1.00 | 1.00 | 0.1525 |

The paper EIIE again collapses to an almost constant, near-fully-invested mix (`weight time-std` ~0.001, turnover 0.0003/bar) and tracks buy-and-hold. It beats equal-weight rebalanced (0.66 vs 0.57) most likely because it trades far less and so pays less commission and bleeds less rebalancing/volatility drag; it does not beat BTC. Active controls (follow-the-winner/loser) are destroyed by the 25 bp commission alone.

## Individual effects at a glance

| # | enhancement | effect vs the unchanged baseline (final wealth 0.662) | verdict |
|---|---|---|---|
| 1 | realistic execution | baseline 0.662 -> 0.657-0.658 (TWAP/VWAP/IS, $10M); active strategies lose far more | costs are real (5-9 bp per $ traded) but the baseline barely trades, so it barely notices |
| 2 | SAC | 0.669 (vs 0.662) | no measurable change: still an almost constant allocation |
| 3 | long/short | free net: 0.602 (cap 1.0), 0.379 (cap 1.5 and 3.0, identical); dollar-neutral: 1.000 (all cash) | no relative-value edge survives costs here; unconstrained net exposure just levers the beta |
| 4 | dynamic universe | dynamic top-6 0.765 vs fixed top-6 0.742, but the equal-weight controls move the other way (0.639 vs 0.672) | no robust effect on returns; matters for tail events, with a lag |
| 5 | tercile attribution | losses concentrate in the low-liquidity tercile for every strategy including passive ones | the pattern is the market, not model skill; active (timing) P&L ~ 0 for EIIE/SAC |

### 1. Execution
* Child-order execution costs about **5-9 bp per dollar traded at $10M** on top of the 25 bp commission
  (EIIE: TWAP 8.0, VWAP 7.4, IS 5.9 bp). The ranking of algorithms is **not consistent**: IS is cheapest for
  EIIE/equal-weight/buy-and-hold and slightly most expensive for 1-day momentum, so no algorithm is a robust winner; differences between them are
  an order of magnitude smaller than the gap to instant filling.
* The baseline barely trades (0.0003 of the book per bar), so realistic execution only moves it from 0.662 to 0.657.
  Strategies that trade matter: with a 2 bp fee, 1-day momentum goes 0.75 -> 0.35 and the cost-blind EIIE 0.53 -> 0.12
  once fills are realistic. **Instant filling at the close is what makes active strategies look viable.**
* Sensitivity: the impact coefficient moves slippage a lot (reversal, TWAP: Y=0.5 -> 18.8% of AUM, Y=2 -> 39.7%); working the order over the
  30 minutes alone (commission-only variant) costs 4.4% of AUM for the 3-hour reversal strategy, i.e. the decision signal is stale by the time it fills.
* Spread and impact are assumptions (no quote data), so the *levels* are scenario outputs; the relative conclusions are what to trust.

### 2. SAC
* In this implementation SAC **does not fix the collapse**: test weights vary even less than the baseline's (time-std 0.0007 vs 0.0013) and
  turnover stays near zero. Validation wealth of later checkpoints *falls* (1.8 -> 1.2-1.3) as the actor drifts to a worse constant mix; the
  validation-selected checkpoint is an early, near-uniform one. This says SAC found no exploitable state dependence in price-only 30-minute
  features under 25 bp costs, not that none exists: a sharper reward, richer features or longer horizons were not tried.

### 3. Long/short
* `ls_free_1.5` and `ls_free_3.0` converge to the **same** constant 134% gross long mix (cash -34%): the cap never binds, and the
  unconstrained policy simply lever the market, which loses in a bear market. Note leverage financing (negative cash) is *not* charged in
  the simulator, so these numbers flatter the levered case. The 1.0 cap run holds 100% long.
* The dollar-neutral model learns to **hold only cash**. Diagnostic (`artifacts/enh/neutral_diag.json`, one seed, 2024 validation): trained
  with no cost term it trades heavily (0.23 of the book per bar, gross 1.1) but is wiped out when charged 25 bp (wealth 0.0001); trained with the cost
  term it goes flat. In this setup no relative-value signal at 30-minute frequency survives a 25 bp commission.

### 4. Dynamic universe
* Both K=6 models beat the 10-coin baseline on test (0.742 / 0.765 vs 0.662; validation only clearly for the dynamic one, 2.24 vs 2.00), but dynamic-vs-fixed is inconsistent between the learned model (+0.023) and the
  equal-weight control (-0.033), so the effect of re-ranking itself is **not established** (2 seeds, one path).
* **Stress test (BNB, -97% in 6 h, delisted):** the fixed-universe model piles into the falling coin (BNB weight rises from 0.16 to 1.00) and ends at 0.075; the models trained on
  re-ranked universes never exceed 15% in it and end at 0.58 (monthly) / 0.56 (weekly). That difference is mostly the *models'* reaction to the crash, not the universe rule alone.
  The re-ranking rule itself is slow: with a 30-day trailing-volume lookback the dead coin stays in the universe for about 20+ days whatever the re-rank frequency
  (weekly and monthly give the same weights after the crash). A faster rule (short lookback or a hard price/volume-collapse exit) was not tested.

### 5. Attribution
* For every strategy - including the passive buy-and-hold and equal-weight ones - most of the loss sits in the low- and mid-liquidity terciles (e.g. EIIE: -2.3% and -1.7% a year,
  versus +1.0% from the high-liquidity tercile; buy-and-hold of each tercile: -46%, -37%, -20%). The identical pattern for the passive controls says this is
  the market's low-liquidity underperformance in 2025-26, not a model effect.
* For EIIE and SAC the **active (timing) component is ~0 bp** versus a passive component of hundreds of bp: the P&L comes from what is held, not from flipping it.
  Only 1-day momentum has a large active component (+2,700 bp in the low-liquidity tercile), and it loses to costs.
* Terciles are only 3/4/3 coins and one bear-market path; there is no evidence here about the professor's monotonicity question beyond this single window.

## 1. Execution realism (TWAP / VWAP / implementation shortfall)

Same trained policies, re-run through each execution algorithm (30 one-minute child orders after the decision; spread + sqrt impact on top of the 25 bp commission). `slippage` = implementation shortfall vs the decision close, in bp per dollar traded. $10M book:

| strategy | execution | final wealth | slippage (% of AUM) | slippage (bp/traded) |
|---|---|---|---|---|
| eiie | instant | 0.6617 | 0.00% | 0.0 |
| eiie | twap | 0.6570 | 0.69% | 8.0 |
| eiie | vwap | 0.6574 | 0.63% | 7.3 |
| eiie | is | 0.6582 | 0.51% | 5.9 |
| ucrp | instant | 0.5743 | 0.00% | 0.0 |
| ucrp | twap | 0.5642 | 1.51% | 3.6 |
| ucrp | vwap | 0.5650 | 1.38% | 3.3 |
| ucrp | is | 0.5662 | 1.19% | 2.8 |
| ubah | instant | 0.6383 | 0.00% | 0.0 |
| ubah | twap | 0.6355 | 0.44% | 43.6 |
| ubah | vwap | 0.6356 | 0.43% | 42.3 |
| ubah | is | 0.6361 | 0.34% | 34.1 |
| momentum_1d | instant | 0.0896 | 0.00% | 0.0 |
| momentum_1d | twap | 0.0464 | 21.79% | 8.0 |
| momentum_1d | vwap | 0.0455 | 22.00% | 8.1 |
| momentum_1d | is | 0.0449 | 23.49% | 8.8 |

**Low-fee regime (2 bp commission, $10M)** so that execution is not drowned by commission:

| strategy | execution | final wealth | Sharpe | slippage (bp/traded) |
|---|---|---|---|---|
| eiie | instant | 0.6776 | -0.03 | 0.0 |
| eiie | twap | 0.6728 | -0.03 | 8.0 |
| eiie | vwap | 0.6732 | -0.03 | 7.3 |
| eiie | is | 0.6740 | -0.03 | 5.9 |
| momentum_1d | instant | 0.7520 | +0.11 | 0.0 |
| momentum_1d | twap | 0.3478 | -0.53 | 8.5 |
| momentum_1d | vwap | 0.3425 | -0.54 | 8.6 |
| momentum_1d | is | 0.3375 | -0.55 | 9.0 |
| reversal_3h | instant | 0.0367 | -2.18 | 0.0 |
| reversal_3h | twap | 0.0001 | -6.94 | 8.2 |
| reversal_3h | vwap | 0.0001 | -7.06 | 8.4 |
| reversal_3h | is | 0.0001 | -6.90 | 8.5 |
| eiie_costblind | instant | 0.5291 | -0.13 | 0.0 |
| eiie_costblind | twap | 0.1175 | -1.38 | 4.9 |
| eiie_costblind | vwap | 0.1137 | -1.40 | 5.0 |
| eiie_costblind | is | 0.1140 | -1.37 | 5.1 |

Sensitivity ($10M, 25 bp fee; slippage % of AUM):

| strategy | variant | slippage % AUM | fee % AUM |
|---|---|---|---|
| reversal_3h | twap, commission only (isolates 30-min delay) | 4.4 | 107.5 |
| reversal_3h | twap, Y=0.5 | 18.8 | 92.6 |
| reversal_3h | twap, Y=1 | 27.2 | 83.7 |
| reversal_3h | twap, Y=2 | 39.7 | 70.6 |
| reversal_3h | is, urgency 1 | 27.1 | 83.9 |
| reversal_3h | is, urgency 3 | 27.3 | 83.6 |
| reversal_3h | is, urgency 6 | 28.8 | 82.0 |
| reversal_3h | twap, fee 10bp | 50.9 | 61.9 |
| eiie_costblind | twap, commission only (isolates 30-min delay) | 3.0 | 104.9 |
| eiie_costblind | twap, Y=0.5 | 11.9 | 96.3 |
| eiie_costblind | twap, Y=1 | 16.8 | 91.6 |
| eiie_costblind | twap, Y=2 | 25.0 | 83.7 |
| eiie_costblind | is, urgency 1 | 16.7 | 91.7 |
| eiie_costblind | is, urgency 3 | 17.7 | 90.7 |
| eiie_costblind | is, urgency 6 | 20.3 | 88.1 |
| eiie_costblind | twap, fee 10bp | 33.6 | 73.3 |

(`eiie_costblind` = EIIE trained with zero cost term, seed 0, an *active* policy used as a stress case; validation selection would have picked a constant seed instead.)

## 2. Soft Actor-Critic

| strategy | final wealth | ann. return | Sharpe | max DD | turnover/bar | gross | net | weight time-std |
|---|---|---|---|---|---|---|---|---|
| EIIE (paper baseline) | 0.662 | -21.0% | -0.05 | 68% | 0.0003 | 0.95 | 0.95 | 0.0013 |
| SAC (Dirichlet policy) | 0.669 | -20.6% | -0.07 | 66% | 0.0008 | 0.91 | 0.91 | 0.0007 |
| equal weight (reference) | 0.574 | -27.2% | -0.13 | 71% | 0.0018 | 1.00 | 1.00 | 0.0000 |

Validation (2024) final wealth by seed - baseline EIIE: 2.00 / 1.80 / 2.00; SAC (chosen checkpoint): 1.80 / 1.90 / 1.24. Training history is in `artifacts/enh/sac_selection.json`.

## 3. Long/short with leverage constraints (8% APR borrow)

| strategy | final wealth | ann. return | Sharpe | max DD | turnover/bar | gross | net | weight time-std |
|---|---|---|---|---|---|---|---|---|
| EIIE long-only (baseline) | 0.662 | -21.0% | -0.05 | 68% | 0.0003 | 0.95 | 0.95 | 0.0013 |
| ls_free_1.0 | 0.602 | -25.2% | -0.08 | 70% | 0.0019 | 1.00 | 1.00 | 0.0006 |
| ls_free_1.5 | 0.379 | -42.6% | -0.15 | 85% | 0.0029 | 1.34 | 1.34 | 0.0000 |
| ls_free_3.0 | 0.379 | -42.6% | -0.15 | 85% | 0.0029 | 1.34 | 1.34 | 0.0000 |
| ls_neutral_1.5 | 1.000 | -0.0% | -2.54 | 0% | 0.0000 | 0.00 | -0.00 | 0.0000 |


## 4. Dynamic universe (K = 6 of 10, monthly re-rank on trailing volume)

| strategy | final wealth | ann. return | Sharpe | max DD | turnover/bar | gross | net | weight time-std |
|---|---|---|---|---|---|---|---|---|
| EIIE, all 10 coins (baseline) | 0.662 | -21.0% | -0.05 | 68% | 0.0003 | 0.95 | 0.95 | 0.0013 |
| EIIE fixed top-6 | 0.742 | -15.7% | +0.02 | 66% | 0.0006 | 0.99 | 0.99 | 0.0014 |
| EIIE dynamic top-6 | 0.765 | -14.2% | -0.03 | 59% | 0.0003 | 0.86 | 0.86 | 0.0107 |
| equal-weight fixed top-6 (control) | 0.672 | -20.4% | -0.07 | 68% | 0.0017 | 1.00 | 1.00 | 0.0000 |
| equal-weight dynamic top-6 (control) | 0.639 | -22.6% | -0.11 | 66% | 0.0016 | 1.00 | 1.00 | 0.0117 |

Membership changed 30 times 2021-2024 and 4 times in the test window (near-tied coins swap at month boundaries; this pool of 10 liquid coins has no real delisting).


**Delisting stress test** - BNB falls 97% in 6 hours on 2025-06-01 and is delisted:

| universe | scenario | final wealth | max DD |
|---|---|---|---|
| fixed | crash | 0.075 | 96% |
| fixed | no-crash | 0.742 | 66% |
| dyn-monthly | crash | 0.583 | 67% |
| dyn-monthly | no-crash | 0.765 | 59% |
| dyn-weekly | crash | 0.557 | 69% |
| dyn-weekly | no-crash | 0.754 | 59% |


## 5. Liquidity-tercile attribution

Terciles are re-ranked monthly on trailing 30-day USDT volume; test-period composition (fraction of time in the tercile): **low-liquidity**: BNB 0.04, ADA 0.91, DOGE 0.05, AVAX 1.00, LINK 1.00, LTC 1.00; **mid-liquidity**: BNB 0.96, SOL 0.19, XRP 0.81, ADA 0.09, DOGE 0.95; **high-liquidity**: BTC 1.00, ETH 1.00, SOL 0.81, XRP 0.19.

| strategy | tercile | avg gross weight | net contribution / yr | sleeve Sharpe | passive (bp, arithmetic) | active (bp, arithmetic) | tercile EW buy&hold |
|---|---|---|---|---|---|---|---|
| eiie | low-liquidity | 0.38 | -2.35% | -0.08 | -389 | -101 | -46% |
| eiie | mid-liquidity | 0.28 | -1.68% | -0.09 | -76 | -7 | -37% |
| eiie | high-liquidity | 0.28 | +1.04% | +0.06 | +303 | +7 | -20% |
| sac | low-liquidity | 0.36 | -2.46% | -0.08 | -370 | +18 | -46% |
| sac | mid-liquidity | 0.27 | -2.00% | -0.11 | -68 | +2 | -37% |
| sac | high-liquidity | 0.27 | +0.39% | +0.02 | +292 | -3 | -20% |
| ls_free_1.5 | low-liquidity | 0.54 | -6.55% | -0.15 | -547 | +0 | -46% |
| ls_free_1.5 | mid-liquidity | 0.40 | -5.37% | -0.20 | -101 | +0 | -37% |
| ls_free_1.5 | high-liquidity | 0.40 | -2.03% | -0.09 | +430 | +0 | -20% |
| eiie_dyn6 | low-liquidity | 0.00 | -0.08% | -1.51 | -65 | -310 | -46% |
| eiie_dyn6 | mid-liquidity | 0.43 | -2.74% | -0.09 | -40 | -28 | -37% |
| eiie_dyn6 | high-liquidity | 0.43 | +1.39% | +0.06 | +459 | -5 | -20% |
| ucrp | low-liquidity | 0.40 | -4.65% | -0.15 | -407 | +0 | -46% |
| ucrp | mid-liquidity | 0.30 | -3.45% | -0.17 | -75 | -0 | -37% |
| ucrp | high-liquidity | 0.30 | -0.76% | -0.04 | +320 | -0 | -20% |
| ubah | low-liquidity | 0.33 | -3.45% | -0.13 | -103 | -681 | -46% |
| ubah | mid-liquidity | 0.33 | -2.77% | -0.13 | +321 | -558 | -37% |
| ubah | high-liquidity | 0.34 | +1.83% | +0.09 | +363 | -85 | -20% |
| momentum_1d | low-liquidity | 0.38 | -33.44% | -0.85 | -309 | +2687 | -46% |
| momentum_1d | mid-liquidity | 0.30 | -47.10% | -1.80 | +87 | +802 | -37% |
| momentum_1d | high-liquidity | 0.32 | -33.26% | -1.31 | +337 | -425 | -20% |

`passive` = average weight x summed returns (what holding the same average weight earns); `active` = the rest (timing/flipping). They are arithmetic sums of per-bar contributions, so they exclude compounding/volatility drag and do not add up to the net contribution (which is after costs and compounding).
