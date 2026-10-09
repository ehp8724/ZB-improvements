"""Professor-suggested enhancements to the EIIE/PVM crypto portfolio replication.

Each module isolates one enhancement so its effect can be measured against the
unchanged paper baseline:

* ``execution``   -- TWAP / VWAP / implementation-shortfall fills with spread + impact
* ``sac``         -- Soft Actor-Critic (Dirichlet policy) replacing the deterministic EIIE
* ``longshort``   -- signed weights under gross-leverage / net-exposure constraints
* ``universe``    -- periodic liquidity-based universe re-selection with transition costs
* ``attribution`` -- liquidity-tercile P&L / Sharpe decomposition
"""
