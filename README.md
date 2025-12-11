# Quant Alpha Model

A quantitative research project that develops and backtests a multi-factor alpha model for Indian mid-cap equities using regime-switching macroeconomic signals. The project includes data engineering, factor construction, econometric modeling, machine-learning-based cross-sectional return forecasts, portfolio optimization, and a full historical backtest with performance and risk analysis.

## Research

placeholder




## Python Project Management with `uv`

### __Key Files__
- `pyproject.toml` - Dependencies, project metadata, tool configs
- `uv.lock` - Exact dependency versions (make sure to commit this)
- `.python-version` - Python version for this project
- `.venv/` - Virtual environment (do not commit this)

### __Quick Reference__
```bash
uv init                   # Start new project
uv python install         # Install Python version
uv add <package>          # Add dependency
uv run <script>           # Run code
uv sync                   # Install/update dependencies
uv lock                   # Update lock file
uv run python --version   # Check project python version
```

### __Installation__

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### __Starting Your Project__

```bash
# Create a new project
uv init quant-alpha-model

# Or initialize in current directory
uv init
```

- Initializing creates three files:
  - `pyproject.toml` - project configuration
  - `.python-version` - Python version lock
  - `README.md`
  - `.gitignore`

### __Python Version Management__

```bash
# Install a specific Python Version
uv python install 3.13

# Use it for your project
uv python pin 3.13

# List available Python Versions
uv python list

# List installed versions
uv python list --only-installed
```

### __Mangaging Python Dependencies__

```bash
# Add a package
uv add requests
uv add pandas numpy # multiple at once

# Add dev dependencies
uv add --dev pytest black ruff

# Add with version constraints
uv add "django>=4.0, <5.0"

# Remove a package
uv remove requests

# Remove package from dev group
uv remove black --group dev

# Update dependencies
uv lock --upgrade # update lock file
uv sync # sync environment with lock file
```

### __Running Code__
```bash
# Run a Python script
uv run main.py

# Run a module
uv run -m pytest

# Run a command in the virtual environment
uv run python -c "import requests; print(requests.__version__)"

# One-off script execution (without adding to project)
uv run --with httpx -- python script.py
```

### __Virtual Enviroments__
```bash
# Create a venv (usually automatic)
uv venv

# Activate it manually (if needed)
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Sync dependencies to venv
uv sync

# Sync only production dependencies (exclude dev)
us sync --no-dev
```

### __VS Code Integration__
Add to your `.vscode/settings.json`

After running `uv sync`, select the interpreter:
- Press `Ctrl + Shift + P` (`Cmd + Shift + P` on Mac)
- Type "Python: Select Interpreter"
- Choose the one in `.venv`
- You can reload VS Code with `Ctrl + Shift + P` (`Cmd + Shift + P` on Mac) and run "Developer: Reload Window"
- Default interpreter path if pointed just to `.venv` will find the correct python installation on both Windows and macOS/Linux
```json
{
    // Python Interpreter
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv",
    "python.terminal.activateEnvironment": true,
    // Testing
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.autoTestDiscoverOnSaveEnabled": true,
    // Formatting & Linting (using Ruff - the modern standard)
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit",
            "source.fixAll": "explicit"
        },
        "editor.insertSpaces": true,
        "editor.tabSize": 4,
        "editor.detectIndentation": false
    },
    // Type Checking
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "python.analysis.diagnosticSeverityOverrides": {
        "reportUnusedImport": "information",
        "reportUnusedVariable": "information"
    },
    // Editor Settings
    "editor.rulers": [
        88,
        120
    ],
    "editor.bracketPairColorization.enabled": true,
    "editor.guides.bracketPairs": true,
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "files.trimFinalNewlines": true,
    // Files to exclude from explorer
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.ruff_cache": true,
        "**/.mypy_cache": true,
        "**/*.egg-info": true
    },
    // Search exclusions
    "search.exclude": {
        "**/.venv": true,
        "**/node_modules": true,
        "**/__pycache__": true,
        "**/*.pyc": true
    },
    // Terminal
    // Set UV_PYTHON_PREFERENCE to only-managed to ensure the terminal uses the workspace virtual environment
    "terminal.integrated.env.windows": {
        "UV_PYTHON_PREFERENCE": "only-managed"
    },
    "terminal.integrated.env.linux": {
        "UV_PYTHON_PREFERENCE": "only-managed"
    },
    "terminal.integrated.env.osx": {
        "UV_PYTHON_PREFERENCE": "only-managed"
    },
    // If you use Jupyter notebooks
    "jupyter.askForKernelRestart": false,
    "notebook.output.textLineLimit": 500,
    // If you want auto-save
    "files.autoSave": "onFocusChange"
}

```

### __Common Workflows__
#### Starting Development
```bash
uv sync # install all dependencies
uv run python main.py
```
#### Add New Dependency
```bash
uv add some-package
# Will automatically update pyproject.toml, uv.lock, and installs the package
```
#### Running Tests
```bash
uv run pytest
# or
uv run python -m pytest
```
#### Formatting and Linting
```bash
uv add --dev ruff
uv run ruff check .
uv run ruff format .
```
### __Extensions to Install__
```bash
# In VS Code, install these:
# 1. Python (ms-python.python)
# 2. Ruff (charliermarsh.ruff)
# 3. Even Better TOML (tamasfe.even-better-toml) - for pyproject.toml
```



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
