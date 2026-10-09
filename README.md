# ZB-improvements

Enhancements to the EIIE/PVM deep-RL crypto portfolio replication, following the professor's
feedback on the Group 4 presentation. Each enhancement is a separate module and is evaluated
**individually** against the unchanged paper baseline on the Binance 1-minute data in `data/`.

| # | enhancement (professor / Vedant) | module | question answered |
|---|---|---|---|
| 1 | TWAP / VWAP / implementation-shortfall execution with spread + market impact | `execution.py` | how much of the paper's edge survives realistic fills? |
| 2 | Soft Actor-Critic instead of the deterministic EIIE | `train.py`, `networks.py` | does exploration fix the collapse to a constant mix? |
| 3 | Long/short with gross-leverage cap and dollar-neutral option | `networks.py` | does shorting / relative value add anything, at what leverage? |
| 4 | Periodically re-ranked universe + transition costs, delisting stress test | `universe.py` | is the fixed-universe assumption doing work? |
| 5 | Liquidity-tercile P&L / Sharpe attribution, passive vs active | `attribution.py` | does the P&L come from illiquid coins, or from trading at all? |

* Method and assumptions: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
* Results: [`docs/RESULTS.md`](docs/RESULTS.md) (generated tables in `artifacts/enh/*.json`, figures in `artifacts/enh/figures/`)
* Paper code (EIIE, PVM, OSBL, exact eq.-14 accounting) is copied unchanged from
  `deep-rl-pm` into `src/rl_portfolio/`.

## Reproduce

```bash
pip install numpy pandas pyarrow scipy matplotlib pytest torch    # CPU torch is enough
python scripts/run_enhancements.py benchmarks
python scripts/run_enhancements.py baseline          # paper EIIE, 3 seeds
python scripts/run_enhancements.py costblind         # active policy used in the execution study
python scripts/run_enhancements.py exec              # enhancement 1
python scripts/run_enhancements.py sac               # enhancement 2
python scripts/run_enhancements.py ls                # enhancement 3
python scripts/run_enhancements.py universe          # enhancement 4
python scripts/run_enhancements.py stress
python scripts/run_enhancements.py main && python scripts/run_enhancements.py attr   # enhancement 5
python scripts/run_enhancements.py plots
python -m pytest tests/test_enhancements.py
```
The first call builds a minute-grid cache (`artifacts/cache/`, git-ignored; about 3 minutes).
