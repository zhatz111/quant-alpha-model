# quant-alpha-model
 A quantitative research project that develops and backtests a multi-factor alpha model for Indian mid-cap equities using regime-switching macroeconomic signals. The project includes data engineering, factor construction, econometric modeling, machine-learning-based cross-sectional return forecasts, portfolio optimization, and a full historical backtest with performance and risk analysis.

## Folder Structure
```bash
quant-alpha-model/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── LICENSE               # optional
│
├── data/
│   ├── raw/              # unmodified downloaded data
│   ├── processed/        # cleaned + joined datasets
│   └── external/         # macro data, alternative data sources
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_construction.ipynb
│   ├── 03_regime_detection.ipynb
│   ├── 04_alpha_modeling.ipynb
│   ├── 05_portfolio_construction.ipynb
│   └── 06_backtesting_results.ipynb
│
├── src/
│   ├── data/
│   │   ├── download_data.py
│   │   ├── clean_data.py
│   │   ├── merge_macro.py
│   │   └── utils.py
│   │
│   ├── factors/
│   │   ├── momentum.py
│   │   ├── value.py
│   │   ├── quality.py
│   │   ├── volatility.py
│   │   └── microstructure.py
│   │
│   ├── models/
│   │   ├── regime_model.py      # HMM, Markov switching, etc.
│   │   ├── factor_model.py      # cross-sectional regression, ML
│   │   └── bayesian_updates.py  # optional Bayesian layer
│   │
│   ├── portfolio/
│   │   ├── optimization.py      # cvxpy optimizers
│   │   ├── risk.py              # volatility, CVaR, beta, exposure
│   │   └── weighting.py         # ranking, volatility-targeting, etc.
│   │
│   ├── backtesting/
│   │   ├── engine.py            # vectorized backtester logic
│   │   ├── metrics.py           # Sharpe, drawdown, turnover
│   │   ├── transaction_costs.py
│   │   └── execution.py
│   │
│   ├── logging/
│   │   ├── logger.py
│   │   └── logging_config.yaml
│   │
│   └── visualization/
│       ├── plots.py
│       └── dashboard.py         # optional streamlit dashboard
│
├── research/
│   ├── draft_report.md          # editable version
│   ├── final_paper.pdf          # polished version for recruiters
│   └── figures/                 # charts exported for the paper
│
├── config/
│   ├── tickers.yaml             # list of NIFTY midcap tickers
│   ├── factors.yaml             # factor definitions + parameters
│   └── backtest.yaml            # start/end dates, universes, etc.
│
├── logs/                    # <-- actual log output files
│   ├── app.log
│   └── debug.log
│
└── tests/
    ├── test_factor_functions.py
    ├── test_backtester.py
    ├── test_regime_model.py
    └── test_portfolio.py

```