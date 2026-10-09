# Enhancement methodology

Source: professor feedback on the Group 4 presentation of Jiang, Xu & Liang (2017),
*A Deep RL Framework for the Financial Portfolio Management Problem* (EIIE/PVM/OSBL),
reproduced on Binance spot data (`ehp8724/deep-rl-pm`).

Every enhancement is changed **one at a time** against the same baseline so its effect
can be read in isolation. Nothing is stacked.

## Common setup

| item | value |
|---|---|
| data | Binance spot 1-minute klines, 10 USDT pairs (BTC ETH BNB SOL XRP ADA DOGE AVAX LINK LTC), Jan 2021 - Sep 2026 |
| decision grid | 30-minute bars; the decision at bar close `t` uses bars `<= t` only |
| splits | train 2021-01 -> 2023-12, validation 2024 (seed/checkpoint selection only), **test 2025-01 -> 2026-09 (30.6k bars)** |
| baseline policy | paper EIIE-CNN (f1=2, f2=20, window 50, previous-weight input, softmax with cash), OSBL + PVM, 80k steps, Adam 3e-5, 3 seeds, seed picked on validation Sharpe |
| baseline accounting | fill at the decision-bar close, flat linear 25 bp commission per side, no slippage (exactly the paper/presentation assumption) |
| engine | one dollar-accounting simulator for every strategy (`src/enhancements/execution.py`), so differences come from the strategy, not the accounting |

Training loss is the paper's `-mean log(w.y * (1 - c*turnover))`. One deliberate detail:
turnover sums over **risky assets only** (cash trades are free), which is what the exact
eq. 14 remainder factor and the simulator both charge.

## 1. Execution realism (TWAP / VWAP / implementation shortfall)

The decision is taken at the close of bar `t`; the order is then **worked over the next 30
one-minute bars** instead of filling instantly at that close.

* TWAP: equal child orders.
* VWAP: child sizes proportional to the average per-minute USDT volume in the same
  30-minute slot over the previous 7 days (known at decision time, no look-ahead).
* Implementation shortfall: Almgren-Chriss style front-loaded trajectory
  `x(k) = sinh(kappa (N-k)) / sinh(kappa N)`, urgency `kappa N = 3` (swept 1/3/6).
* Each child fill pays the **same flat 25 bp commission as the baseline** plus
  `half_spread + Y * sigma_1m * sqrt(child_notional / minute_volume)`
  (square-root impact, `Y = 1`, `sigma_1m` = trailing 6h minute volatility, participation capped at 1).
  Half-spreads (0.1-1.5 bp by coin) and `Y` are scenario assumptions: candle data contain no quotes.
* Reported at $1M and $10M books. Implementation shortfall vs the decision close is split
  out as `slippage`, so delay drift, spread and impact are all inside it; a
  "commission only" variant isolates the pure 30-minute delay.
* The same trained policies are re-run through each algorithm; the policy is not retrained.

## 2. Soft Actor-Critic

Replaces the deterministic EIIE policy by SAC on the portfolio MDP.

* state: price tensor (3, K, 50) + previous action; action: simplex weights.
* actor: EIIE CNN trunk -> scores -> `Dirichlet(kappa * softmax(scores))`; `kappa` (total concentration) is learned, so the policy can be exploratory or sharp.
* critics: twin permutation-equivariant EIIE-style Q networks + Polyak targets.
* reward: net log growth `log(a.y_next * (1 - c*turnover))` x1000, `gamma = 0.9`.
* automatic entropy temperature toward the entropy of a reference Dirichlet (concentration 120), i.e. explicit exploration, which the professor's explore/exploit point asks for.
* 300k environment steps (16 parallel 64-bar segments from random history windows), replay buffer, 64 updates per rollout; checkpoint chosen on validation Sharpe; deterministic (mean) action at test.

## 3. Long/short with leverage constraints

EIIE trunk -> `a_i = tanh(score_i)`; optional cross-sectional demeaning (**dollar-neutral**
relative value, "long BTC / short ETH"); weights are scaled **down only** so that gross
exposure `sum|w_i| <= L`. Cash is the residual, so shorting finances longs. Variants:
`L = 1.0, 1.5, 3.0` with free net exposure, and `L = 1.5` dollar-neutral.
Shorts pay 8% APR borrow (scenario) in both the training loss and the simulator.
The gross cap is the guard against the "10x" leverage the professor warned about; no
liquidation/margin model is simulated, so results at `L = 3` are optimistic.

## 4. Dynamic universe

The paper fixes the coins and admits picking them with look-ahead. Here a K = 6 slot model
is used (EIIE's evaluator is shared across assets, so it applies to any 6-subset):

* **fixed-6**: top 6 by trailing-30d USDT volume at the end of training, frozen afterwards.
* **dynamic-6**: re-ranked on the first bar of each month using trailing volume only.
* PVM, rewards and costs live in the full 10-coin space, so a coin that drops out is
  actually sold (the "transition problem") and an entering coin is bought, both paying costs.
* Both have equal-weight controls on the same universes (`ucrp_fixed6`, `ucrp_dyn6`).
* **Delisting stress test**: the least-liquid held coin crashes 97% within 6 hours on
  2025-06-01 and is delisted (no volume). Fixed universe vs monthly vs weekly re-ranking.
  This is synthetic and meant to show mechanics (how long the model keeps holding a dead coin).

## 5. Liquidity-tercile attribution

No market-cap field exists, so trailing 30-day USDT volume is the size/liquidity proxy; coins
are bucketed monthly (3 / 4 / 3 coins) from past data only. For every strategy, per tercile:
net P&L and share, annualised contribution, sleeve Sharpe, cost, gross weight, and a
**passive vs active split**: `passive = mean_weight x sum of returns` (what buy-and-hold of the
same average weight would earn), `active = remainder` (earned by timing/flipping).
This answers "is the P&L from low-cap names, and is it from holding or flipping?".

## Honest limitations

* Candle data only: spreads, queue position and depth are assumed, not observed.
* The 10-coin pool contains no true delistings, so the dynamic universe is exercised by
  monthly liquidity ranking and a synthetic crash.
* Test-period results are one path; seeds are few (2-3). Validation = 2024 only.
* No margin call / liquidation model for the long/short cases.
