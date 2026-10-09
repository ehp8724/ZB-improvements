#!/usr/bin/env python
"""Render docs/RESULTS.md from artifacts/enh/*.json (tables only come from saved results)."""
import json
from pathlib import Path

D = Path("artifacts/enh")
J = lambda n: json.loads((D / n).read_text())  # noqa: E731
main = {r["strategy"]: r for r in J("main_table.json")}
att = J("attribution.json")
ex = J("exec_table.json"); low = J("exec_lowfee.json"); sens = J("exec_sensitivity.json")
stress = J("stress_delisting.json"); chg = J("universe_changes.json")


def tbl(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def mrow(n, label=None):
    r = main[n]
    return [label or n, f"{r['fapv']:.3f}", f"{r['ann_return_pct']:+.1f}%", f"{r['sharpe']:+.2f}",
            f"{r['max_drawdown_pct']:.0f}%", f"{r['turnover_per_bar']:.4f}", f"{r['avg_gross']:.2f}",
            f"{r['avg_net']:.2f}", f"{r['weight_time_std']:.4f}"]


MH = ["strategy", "final wealth", "ann. return", "Sharpe", "max DD", "turnover/bar", "gross", "net", "weight time-std"]
parts = ["# Results\n",
 "Test period 2025-01-01 -> 2026-09-30 (30,624 half-hour decisions), $1M book, paper accounting "
 "(instant fill at the bar close, flat 25 bp) unless a section says otherwise. "
 "Method and assumptions: [METHODOLOGY.md](METHODOLOGY.md). All numbers are generated from "
 "`artifacts/enh/*.json` by `scripts/make_results_md.py`.\n",
 "**Context that matters for every number below:** the test window is a bear market "
 "(equal-weight buy-and-hold ended at 0.64, BTC at 0.89, only the hindsight best coin made money). "
 "Each method row is **one seed, chosen on 2024 validation Sharpe** (2-3 seeds trained), "
 "so differences of a few percent are within seed noise.\n",
 "## Baseline and benchmarks\n",
 tbl(MH, [mrow(n) for n in ["eiie", "ubah", "ucrp", "btc_hold", "best_stock_hindsight", "momentum_1d", "reversal_3h"]]),
 "\nThe paper EIIE again collapses to an almost constant, near-fully-invested mix "
 "(`weight time-std` ~0.001, turnover 0.0003/bar) and tracks buy-and-hold. It beats equal-weight "
 "rebalanced (0.66 vs 0.57) most likely because it trades far less and so pays less commission and bleeds less "
 "rebalancing/volatility drag; it does not beat BTC. Active controls (follow-the-winner/loser) are destroyed by the 25 bp commission alone.\n",
]

FINDINGS = """## Individual effects at a glance

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
"""
parts += [FINDINGS]

# ---- 1 execution
parts += ["## 1. Execution realism (TWAP / VWAP / implementation shortfall)\n",
 "Same trained policies, re-run through each execution algorithm (30 one-minute child orders after the decision; "
 "spread + sqrt impact on top of the 25 bp commission). `slippage` = implementation shortfall vs the decision close, in "
 "bp per dollar traded. $10M book:\n"]
rows = []
for s in ["eiie", "ucrp", "ubah", "momentum_1d"]:
    for mode in ["instant", "twap", "vwap", "is"]:
        r = next(x for x in ex if x["strategy"] == s and x["mode"] == mode and x["aum"] == 1e7)
        rows.append([s, mode, f"{r['fapv']:.4f}", f"{r['slippage_pct_of_aum']:.2f}%", f"{r['slippage_bps_per_traded']:.1f}"])
parts += [tbl(["strategy", "execution", "final wealth", "slippage (% of AUM)", "slippage (bp/traded)"], rows),
 "\n**Low-fee regime (2 bp commission, $10M)** so that execution is not drowned by commission:\n"]
rows = []
for s in ["eiie", "momentum_1d", "reversal_3h", "eiie_costblind"]:
    for mode in ["instant", "twap", "vwap", "is"]:
        r = next(x for x in low if x["strategy"] == s and x["mode"] == mode and x["aum"] == 1e7)
        rows.append([s, mode, f"{r['fapv']:.4f}", f"{r['sharpe']:+.2f}", f"{r['slippage_bps_per_traded']:.1f}"])
parts += [tbl(["strategy", "execution", "final wealth", "Sharpe", "slippage (bp/traded)"], rows),
 "\nSensitivity ($10M, 25 bp fee; slippage % of AUM):\n",
 tbl(["strategy", "variant", "slippage % AUM", "fee % AUM"],
     [[x["strategy"], x["variant"], f"{x['slippage_pct_of_aum']:.1f}", f"{x['fee_pct_of_aum']:.1f}"] for x in sens]),
 "\n(`eiie_costblind` = EIIE trained with zero cost term, seed 0, an *active* policy used as a stress case; "
 "validation selection would have picked a constant seed instead.)\n"]

# ---- 2 SAC
parts += ["## 2. Soft Actor-Critic\n",
 tbl(MH, [mrow("eiie", "EIIE (paper baseline)"), mrow("sac", "SAC (Dirichlet policy)"), mrow("ucrp", "equal weight (reference)")]),
 "\nValidation (2024) final wealth by seed - baseline EIIE: 2.00 / 1.80 / 2.00; SAC (chosen checkpoint): 1.80 / 1.90 / 1.24. "
 "Training history is in `artifacts/enh/sac_selection.json`.\n"]

# ---- 3 LS
parts += ["## 3. Long/short with leverage constraints (8% APR borrow)\n",
 tbl(MH, [mrow("eiie", "EIIE long-only (baseline)")] + [mrow(n) for n in
     ["ls_free_1.0", "ls_free_1.5", "ls_free_3.0", "ls_neutral_1.5"]]), "\n"]

# ---- 4 universe
tests = [c for c in chg if c["time"] >= "2025-01-01"]
parts += ["## 4. Dynamic universe (K = 6 of 10, monthly re-rank on trailing volume)\n",
 tbl(MH, [mrow("eiie", "EIIE, all 10 coins (baseline)"), mrow("eiie_fixed6", "EIIE fixed top-6"), mrow("eiie_dyn6", "EIIE dynamic top-6"),
          mrow("ucrp_fixed6", "equal-weight fixed top-6 (control)"), mrow("ucrp_dyn6", "equal-weight dynamic top-6 (control)")]),
 f"\nMembership changed {len(chg)} times 2021-2024 and {len(tests)} times in the test window "
 "(near-tied coins swap at month boundaries; this pool of 10 liquid coins has no real delisting).\n",
 f"\n**Delisting stress test** - {stress['asset']} falls 97% in 6 hours on {stress['when']} and is delisted:\n",
 tbl(["universe", "scenario", "final wealth", "max DD"],
     [[r["universe"], r["scenario"], f"{r['fapv']:.3f}", f"{r['max_drawdown_pct']:.0f}%"] for r in stress["rows"]]), "\n"]

# ---- 5 attribution
parts += ["## 5. Liquidity-tercile attribution\n",
 "Terciles are re-ranked monthly on trailing 30-day USDT volume; test-period composition (fraction of time in the tercile): "
 + "; ".join(f"**{k}**: " + ", ".join(f"{s} {v:.2f}" for s, v in c.items()) for k, c in att["composition"].items()) + ".\n"]
rows = []
for s in ["eiie", "sac", "ls_free_1.5", "eiie_dyn6", "ucrp", "ubah", "momentum_1d"]:
    for r in att["tables"][s]:
        rows.append([s, r["bucket"], f"{r['avg_gross_weight']:.2f}", f"{r['ann_contribution_pct']:+.2f}%",
                     f"{r['sleeve_sharpe']:+.2f}", f"{r['gross_passive_bps_total']:+.0f}", f"{r['gross_active_bps_total']:+.0f}",
                     f"{r['bucket_ew_bh_return_pct']:+.0f}%"])
parts += [tbl(["strategy", "tercile", "avg gross weight", "net contribution / yr", "sleeve Sharpe",
               "passive (bp, arithmetic)", "active (bp, arithmetic)", "tercile EW buy&hold"], rows),
 "\n`passive` = average weight x summed returns (what holding the same average weight earns); `active` = the rest "
 "(timing/flipping). They are arithmetic sums of per-bar contributions, so they exclude compounding/volatility drag and "
 "do not add up to the net contribution (which is after costs and compounding).\n"]
Path("docs/RESULTS.md").write_text("\n".join(parts))
print("docs/RESULTS.md written")
